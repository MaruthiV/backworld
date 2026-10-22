import argparse
import hashlib
import json
import os
import signal
import time
import types
import uuid
from pathlib import Path

from service_operator import episode, hotel, http_probe, kubernetes, pilot, relay, retry, retry_episode, selector, tools, workflow
from service_operator.kubernetes import LocalCluster, save
from service_retry_audit import INPUTS, KUBECTL_SHA256, stop_workload
from service_retry_episode_audit import normalize_spec, public_script
from service_selector_audit import InjectorAdapter


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = LocalCluster(output, args.kind)
    started = time.monotonic()
    summary = {"status": "running", "run_id": runtime.name, "cleanup_ok": False}
    images = {**retry.images(args.availability_build), "client": hotel.CLIENT_IMAGE}
    kubectl = args.inputs.resolve() / "bin/kubectl"
    require(hashlib.sha256(kubectl.read_bytes()).hexdigest() == KUBECTL_SHA256, "kubectl differs")
    init_scripts = {}
    for name in ("geo", "rate"):
        relative = f"applications/hotelReservation/k8s-{name}-mongo.sh"
        body = (args.inputs / relative).read_bytes()
        require(hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == INPUTS[relative], "initialization source differs")
        init_scripts[name] = body.decode()
    functions = selector.load_selector_functions(args.injector)
    sources = [Path(__file__), Path(__file__).with_name("service_retry_audit.py"),
               Path(__file__).with_name("service_retry_episode_audit.py"), Path(__file__).with_name("service_selector_audit.py"),
               args.workload, args.injector, args.availability_build, Path("patches/hotel-reservation-availability.patch"),
               *[args.inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh" for name in ("geo", "rate")],
               *[Path(module.__file__) for module in (episode, hotel, http_probe, kubernetes, pilot, relay, retry, retry_episode, selector, tools, workflow)]]
    (output / "source").mkdir()
    hashes = {}
    for path in sources:
        body = path.read_bytes()
        (output / "source" / path.name).write_bytes(body)
        hashes[path.name] = hashlib.sha256(body).hexdigest()
    (output / "protocol.txt").write_bytes(Path("docs/SERVICE_PILOT_PROTOCOL.md").read_bytes())
    save(output / "protocol.json", {"version": 2, "scope": "shared tools/transport audit, not independent pilot episodes or a model result",
         "images": images, "source_sha256": hashes, "node_image": kubernetes.NODE_IMAGE,
         "kind_sha256": kubernetes.KIND_SHA256, "kubectl_sha256": KUBECTL_SHA256,
         "public_task": pilot.public_task(), "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048}, "overall_limit_s": 1200})
    original_env = {key: os.environ.get(key) for key in ("PATH", "KUBECONFIG")}
    workload = None
    namespace, private_namespace = "app-" + uuid.uuid4().hex[:10], "observer-" + uuid.uuid4().hex[:10]
    summary.update({"namespace": namespace, "private_namespace": private_namespace})

    def route(name, expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = episode.routing_state(runtime, namespace)
            save(output / (name + "-routing.json"), state)
            if selector.check_route(state, expected):
                return
            time.sleep(.5)
        raise RuntimeError("frontend endpoint state did not converge")

    def resources(name):
        values = {key: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + key])
                  for key in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
        save(output / (name + "-resources.json"), values)
        require(dict(line.split() for line in values["memory.events"].splitlines())["oom_kill"] == "0", "node OOM kill")

    try:
        save(output / "unrelated-before.json", [json.loads(line) for line in runtime.command(runtime.docker + ["ps", "--format", "json"]).splitlines()])
        runtime.start()
        runtime.load_images(images)
        os.environ["PATH"] = str(kubectl.parent) + os.pathsep + (original_env["PATH"] or "")
        os.environ["KUBECONFIG"] = str(runtime.kubeconfig)
        runtime.apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}} for name in (namespace, private_namespace)])
        objects = retry.profile(namespace, runtime.aliases, init_scripts)
        client = pilot.client_objects(private_namespace, namespace, runtime.aliases["client"])
        save(output / "application.json", objects)
        save(output / "private-client.json", client)
        role = tools.operator_role(namespace, "pilot")
        save(output / "public-role.json", role)
        runtime.apply([*objects, *client, *retry_episode.metric_services(namespace), *role])
        deadline = time.monotonic() + 180
        for obj in [*objects, *client]:
            if obj["kind"] == "Deployment":
                remaining = max(1, int(deadline - time.monotonic()))
                require(time.monotonic() < deadline, "rollout deadline")
                runtime.kube(["-n", obj["metadata"]["namespace"], "rollout", "status", "deployment/" + obj["metadata"]["name"], f"--timeout={remaining}s"], timeout=remaining + 10)
        binary = runtime.kube(["-n", namespace, "exec", "deployment/reservation", "--", "sha256sum", "/go/bin/reservation"])
        save(output / "reservation-binary.json", {"sha256sum": binary})
        require(binary.split()[0] == images["app"]["binary_sha256"], "runtime binary differs")
        rows = json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-profile", "--", "mongo", "--quiet", "--eval", retry.CATALOG_JS]))
        catalog = {row["id"]: row for row in rows}
        save(output / "catalog.json", rows)
        require(len(catalog) == 80, "catalog incomplete")
        recorder = retry.RecordedRequests(output / "http.jsonl", identify=True)
        workload_class = retry.load_workload(args.workload, recorder)
        workload = workload_class(namespace)
        forward_class = type(workload.frontend)
        workload.frontend.stop()
        workload.frontend = forward_class(private_namespace, "client", 8080)
        original_start = workload.frontend.start

        def start_unless_stopping(*a, **kw):
            if workload._stop.is_set():
                raise ValueError("workload is stopping")
            return original_start(*a, **kw)

        workload.frontend.start = start_unless_stopping
        workload.frontend.start()
        ledger = retry_episode.SearchLedger(runtime, namespace, workload, output / "private-ledger")
        require(hotel.fresh_seed(ledger.before), "seed differs")
        warmup, deadline = [], time.monotonic() + 20
        while time.monotonic() < deadline:
            response = http_probe.request(f"http://127.0.0.1:{workload.frontend.local_port}" + workload._PATH, 4)
            warmup.append(response)
            save(output / "warmup.json", warmup)
            try:
                retry.select_hotel(response, catalog)
                break
            except ValueError:
                time.sleep(.5)
        else:
            raise RuntimeError("search warmup deadline")
        ledger.probe("baseline")
        require(ledger.evaluate(catalog, "baseline")["final_valid"], "baseline booking invalid")
        backend = retry_episode.RetryBackend(runtime, namespace)
        checks = []
        for verb, resource, name, expected in [
            ("patch", "services", "frontend", True), ("patch", "services", "search", True), ("patch", "services", "rate", True),
            ("patch", "services", "mongodb-reservation", False), ("patch", "services", "search-metrics", False),
            ("patch", "deployments.apps", "rate", True), ("patch", "deployments.apps", "mongodb-rate", False),
            ("get", "secrets", "anything", False), ("delete", "deployments.apps", "frontend", False)]:
            answer = backend.kube(["auth", "can-i", verb, resource + "/" + name], check=False)
            checks.append({"verb": verb, "resource": resource, "name": name, "expected": expected, "answer": answer})
        runtime.kube([*backend.identity, "-n", private_namespace, "get", "pods", "-o", "json"], check=False)
        denied = json.loads((output / f"command-{runtime.count:04d}.json").read_text())
        denied_text = (output / f"command-{runtime.count:04d}.log").read_text()
        save(output / "permissions.json", {"checks": checks, "actual_private_deny": denied["returncode"] == 1 and "Forbidden" in denied_text})
        require(all(row["answer"] == ("yes" if row["expected"] else "no") for row in checks) and denied["returncode"] == 1 and "Forbidden" in denied_text, "pilot permissions differ")
        for name in ("search", "rate"):
            save(output / (name + "-public-metrics.json"), backend.perform("metrics", {"service": name}))
        before = json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))
        save(output / "before-deployments.json", before)

        def measure(name, seconds=20):
            a = workload.metrics.snapshot()
            t0 = time.monotonic()
            time.sleep(seconds)
            t1 = time.monotonic()
            b = workload.metrics.snapshot()
            time.sleep(4.25)
            sample = retry.window(recorder.snapshot(), t0, t1, catalog, a, b)
            save(output / (name + ".json"), {"start_s": t0, "end_s": t1, "before": a, "after": b, "sample": sample})
            print(json.dumps({"phase": name, "sample": sample}), flush=True)
            resources(name)
            return sample

        def burst(name):
            t0 = time.monotonic()
            workload.run_trigger(40, 10)
            t1 = time.monotonic()
            time.sleep(4.25)
            cohort = [e for e in recorder.snapshot() if e["kind"] == "search" and t0 <= e["started_s"] < t1]
            save(output / (name + ".json"), {"start_s": t0, "end_s": t1, "actual_started": len(cohort)})
            require(len(cohort) / (t1 - t0) >= 32 and all("finished_s" in e for e in cohort), "burst undelivered")

        def actor(name):
            def record(event):
                with (output / (name + "-public.jsonl")).open("a") as handle:
                    handle.write(json.dumps(event) + "\n")

            context = pilot.public_task()
            save(output / (name + "-task.json"), context)
            public = tools.PublicTools(backend, ledger.public_check, record, time.monotonic(), profile="pilot", deadline_s=300, max_calls=64)
            return public

        route("healthy", ["frontend"])
        workload.start()
        time.sleep(20)
        require(retry.healthy(measure("healthy"), initial=True), "relay healthy load invalid")
        adapter = InjectorAdapter(runtime, namespace, output / "injector-actions.jsonl")
        functions["inject_service_wrong_pod_selection"](types.SimpleNamespace(kubectl=adapter, namespace=namespace), ["frontend", "search"])
        route("selector-fault", ["frontend", "search"])
        sample = measure("selector-fault")
        require(6.4 <= sample["actual_rate"] <= 9.6 and sample["completed"] == sample["actual_started"]
                and .2 <= sample["success_rate"] <= .8, "relay did not expose partial selector failure")
        public = actor("selector")
        for call in [{"tool": "inspect", "arguments": {"resource": "endpoints", "name": "frontend"}},
                     {"tool": "set_selector", "arguments": {"service": "frontend", "selector": {"io.kompose.service": "frontend"}}},
                     {"tool": "finish", "arguments": {"summary": "routing corrected"}}]:
            require(public.invoke(call)["ok"], "selector public action failed")
        route("selector-repaired", ["frontend"])
        require(retry.healthy(measure("selector-repaired")), "selector repair failed")
        ledger.probe("after-selector")
        require(ledger.evaluate(catalog, "selector")["final_valid"], "selector booking failed")
        burst("trigger")
        time.sleep(20)
        require(retry.collapsed(measure("retry-fault", 25)), "relay did not sustain retry collapse")
        public = actor("retry")
        public_script(public, "repair")
        require(public.finished_at is not None, "retry actor did not finish")
        time.sleep(20)
        require(retry.healthy(measure("retry-repaired")), "retry repair failed through relay")
        burst("replay-trigger")
        time.sleep(20)
        require(retry.healthy(measure("replay")), "retry replay failed through relay")
        ledger.probe("final")
        stop_workload(workload)
        evaluation = ledger.evaluate(catalog)
        require(evaluation["invariants_preserved"] and evaluation["baseline_valid"] and evaluation["final_valid"], "stored workflow invalid")
        after = json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))
        save(output / "after-deployments.json", after)
        baseline = {obj["metadata"]["name"]: normalize_spec(obj) for obj in before["items"]}
        final = {obj["metadata"]["name"]: normalize_spec(obj) for obj in after["items"]}
        for name in ("frontend", "search"):
            labels = final[name]["template"]["metadata"]["labels"]
            require(labels.pop("service-route", None) == "frontend", "unexpected injector labels")
        require(baseline == final and len(final) == 17, "unrelated deployment fields changed")
        require(all(d["status"].get("replicas") == d["status"].get("updatedReplicas") == d["status"].get("readyReplicas") == 1
                    and d["status"].get("observedGeneration", 0) >= d["metadata"]["generation"] for d in after["items"]), "final topology not current")
        route("final", ["frontend"])
        relay_log = runtime.kube(["-n", private_namespace, "logs", "deployment/client"])
        (output / "relay.jsonl").write_text(relay_log + "\n")
        save(output / "http-events.json", recorder.snapshot())
        with workload._lock:
            save(output / "driver.json", {"submissions": list(workload._submissions), "outcomes": list(workload._events)})
            require(len(workload._submissions) == len(workload._events), "driver lost outcomes")
        require((output / "selector-task.json").read_bytes() == (output / "retry-task.json").read_bytes(), "case-specific public bootstrap")
        summary.update({"status": "passed", "confirmed_bookings": evaluation["confirmed_bookings"]})
    except BaseException as exc:
        summary.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        signal.alarm(0)
        if workload is not None:
            try:
                stop_workload(workload)
                (output / "partial-relay.jsonl").write_text(runtime.kube(["-n", private_namespace, "logs", "deployment/client"], check=False) + "\n")
                save(output / "partial-http-events.json", recorder.snapshot())
            except Exception as exc:
                summary["driver_cleanup_error"] = str(exc)
        summary["cleanup_ok"] = runtime.close()["ok"]
        if not summary["cleanup_ok"]:
            summary["status"] = "failed"
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        summary["elapsed_s"] = time.monotonic() - started
        save(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=Path, default=Path("results/service-kube-runtime-20261021/kind"))
    parser.add_argument("--inputs", type=Path, default=Path("results/service-retry-source-20261021"))
    parser.add_argument("--workload", type=Path, default=Path("results/service-plan-source-20261019/sregym/sregym/generators/workload/hotel_search.py"))
    parser.add_argument("--injector", type=Path, default=Path("results/service-kube-runtime-20261021/inject_virtual.py"))
    parser.add_argument("--availability-build", type=Path, default=Path("results/service-availability-build-20261022-02/build.json"))
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(1200)
    run(args)


if __name__ == "__main__":
    main()
