import argparse
import copy
import hashlib
import json
import os
import signal
import time
import types
import uuid
from pathlib import Path

from service_operator import episode, hotel, http_probe, kubernetes, pilot, relay, retry, retry_episode, runbook, selector, tools, workflow
from service_operator.kubernetes import LocalCluster, save
from service_pilot_audit import require
from service_retry_audit import INPUTS, KUBECTL_SHA256, stop_workload
from service_retry_episode_audit import audit_permissions, normalize_spec
from service_selector_audit import InjectorAdapter


CONDITIONS = (("healthy", "runbook"), ("selector", "runbook"),
              ("retry", "runbook"), ("selector", "premature_finish"))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = LocalCluster(output, args.kind)
    started = time.monotonic()
    summary = {"status": "running", "run_id": runtime.name, "episodes": [], "cleanup_ok": False}
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
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ("service_pilot_audit.py", "service_retry_audit.py", "service_retry_episode_audit.py", "service_selector_audit.py")],
               args.workload, args.injector, args.availability_build, Path("patches/hotel-reservation-availability.patch"),
               *[args.inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh" for name in ("geo", "rate")],
               *[Path(module.__file__) for module in
                 (episode, hotel, http_probe, kubernetes, pilot, relay, retry, retry_episode, runbook, selector, tools, workflow)]]
    (output / "source").mkdir()
    hashes = {}
    for path in sources:
        body = path.read_bytes()
        (output / "source" / path.name).write_bytes(body)
        hashes[path.name] = hashlib.sha256(body).hexdigest()
    (output / "protocol.txt").write_bytes(Path("docs/SERVICE_PILOT_EPISODE_PROTOCOL.md").read_bytes())
    (output / "manifest.txt").write_bytes(Path("docs/SERVICE_PILOT_MANIFEST.md").read_bytes())
    save(output / "protocol.json", {"version": 1, "scope": "fresh development cases and blind app-aware runbook; no model or learning result",
         "conditions": CONDITIONS, "public_task": pilot.public_task(), "source_sha256": hashes, "images": images,
         "node_image": kubernetes.NODE_IMAGE, "kind_sha256": kubernetes.KIND_SHA256, "kubectl_sha256": KUBECTL_SHA256,
         "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048}, "overall_limit_s": 1800})
    original_env = {key: os.environ.get(key) for key in ("PATH", "KUBECONFIG")}
    workload = recorder = folder = private_namespace = None
    prior_ids = set()

    def resources(path):
        values = {key: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + key])
                  for key in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
        save(path, values)
        require(dict(line.split() for line in values["memory.events"].splitlines())["oom_kill"] == "0", "node OOM kill")

    try:
        save(output / "unrelated-before.json", [json.loads(line) for line in runtime.command(runtime.docker + ["ps", "--format", "json"]).splitlines()])
        runtime.start()
        runtime.load_images(images)
        os.environ["PATH"] = str(kubectl.parent) + os.pathsep + (original_env["PATH"] or "")
        os.environ["KUBECONFIG"] = str(runtime.kubeconfig)
        for index, (case, actor) in enumerate(CONDITIONS, 1):
            folder = output / f"episode-{index}"
            folder.mkdir()
            namespace, private_namespace = "app-" + uuid.uuid4().hex[:10], "observer-" + uuid.uuid4().hex[:10]
            entry = {"case": case, "actor": actor, "namespace": namespace, "private_namespace": private_namespace}
            summary["episodes"].append(entry)
            save(output / "summary.json", summary)
            print(json.dumps({"episode": index, "case": case, "actor": actor}), flush=True)
            runtime.apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}} for name in (namespace, private_namespace)])
            objects = retry.profile(namespace, runtime.aliases, init_scripts)
            client = pilot.client_objects(private_namespace, namespace, runtime.aliases["client"])
            save(folder / "application.json", objects)
            save(folder / "private-client.json", client)
            runtime.apply([*objects, *client, *retry_episode.metric_services(namespace), *tools.operator_role(namespace, "pilot")])
            deadline = time.monotonic() + 180
            for obj in [*objects, *client]:
                if obj["kind"] == "Deployment":
                    remaining = max(1, int(deadline - time.monotonic()))
                    require(time.monotonic() < deadline, "rollout deadline")
                    runtime.kube(["-n", obj["metadata"]["namespace"], "rollout", "status", "deployment/" + obj["metadata"]["name"], f"--timeout={remaining}s"], timeout=remaining + 10)
            binary = runtime.kube(["-n", namespace, "exec", "deployment/reservation", "--", "sha256sum", "/go/bin/reservation"])
            save(folder / "reservation-binary.json", {"sha256sum": binary})
            require(binary.split()[0] == images["app"]["binary_sha256"], "runtime binary differs")
            rows = json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-profile", "--", "mongo", "--quiet", "--eval", retry.CATALOG_JS]))
            catalog = {row["id"]: row for row in rows}
            save(folder / "catalog.json", rows)
            require(len(catalog) == 80, "catalog incomplete")
            recorder = retry.RecordedRequests(folder / "http.jsonl", identify=True)
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
            ledger = retry_episode.SearchLedger(runtime, namespace, workload, folder / "private-ledger")
            ids = {row["_id"] for row in ledger.before["reservations"]}
            require(hotel.fresh_seed(ledger.before) and not ids.intersection(prior_ids), "reset seed stale or invalid")
            warmup, deadline = [], time.monotonic() + 20
            while time.monotonic() < deadline:
                response = http_probe.request(f"http://127.0.0.1:{workload.frontend.local_port}" + workload._PATH, 4)
                warmup.append(response)
                save(folder / "warmup.json", warmup)
                try:
                    retry.select_hotel(response, catalog)
                    break
                except ValueError:
                    time.sleep(.5)
            else:
                raise RuntimeError("search warmup deadline")
            ledger.probe("baseline")
            require(ledger.evaluate(catalog, "baseline")["final_valid"], "baseline booking invalid")
            baseline_state = ledger.snapshot()
            backend = retry_episode.RetryBackend(runtime, namespace)
            audit_permissions(runtime, backend, private_namespace, folder / "permissions.json")
            selector_permissions = []
            for name, expected in (("frontend", True), ("search", True), ("rate", True), ("mongodb-reservation", False)):
                answer = backend.kube(["auth", "can-i", "patch", "services/" + name], check=False)
                selector_permissions.append({"name": name, "expected": expected, "answer": answer})
            save(folder / "selector-permissions.json", selector_permissions)
            require(all(row["answer"] == ("yes" if row["expected"] else "no") for row in selector_permissions), "selector permissions differ")
            before = json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))
            before_services = json.loads(runtime.kube(["-n", namespace, "get", "services", "-o", "json"]))
            save(folder / "before-deployments.json", before)
            save(folder / "before-services.json", before_services)

            def route(name, expected, wait=True):
                deadline = time.monotonic() + (30 if wait else 0)
                while True:
                    state = episode.routing_state(runtime, namespace)
                    save(folder / (name + "-routing.json"), state)
                    ok = selector.check_route(state, expected)
                    if ok or time.monotonic() >= deadline:
                        return ok
                    time.sleep(.5)

            def measure(name, seconds=20):
                a = workload.metrics.snapshot()
                t0 = time.monotonic()
                time.sleep(seconds)
                t1 = time.monotonic()
                b = workload.metrics.snapshot()
                time.sleep(4.25)
                sample = retry.window(recorder.snapshot(), t0, t1, catalog, a, b)
                save(folder / (name + ".json"), {"start_s": t0, "end_s": t1, "before": a, "after": b, "sample": sample})
                print(json.dumps({"episode": index, "phase": name, "sample": sample}), flush=True)
                resources(folder / (name + "-resources.json"))
                return sample

            def burst(name):
                t0 = time.monotonic()
                workload.run_trigger(40, 10)
                t1 = time.monotonic()
                time.sleep(4.25)
                cohort = [e for e in recorder.snapshot() if e["kind"] == "search" and t0 <= e["started_s"] < t1]
                save(folder / (name + ".json"), {"start_s": t0, "end_s": t1, "actual_started": len(cohort)})
                require(len(cohort) / (t1 - t0) >= 32 and all("finished_s" in e for e in cohort), "burst undelivered")

            require(route("healthy", ["frontend"]), "initial route invalid")
            workload.start()
            time.sleep(20)
            require(retry.healthy(measure("healthy"), initial=True), "healthy load invalid")
            require(json.dumps(ledger.snapshot(), sort_keys=True) == json.dumps(baseline_state, sort_keys=True), "search changed state")
            if case == "selector":
                adapter = InjectorAdapter(runtime, namespace, folder / "injector-actions.jsonl")
                functions["inject_service_wrong_pod_selection"](types.SimpleNamespace(kubectl=adapter, namespace=namespace), ["frontend", "search"])
                require(route("fault", ["frontend", "search"]), "selector injection did not converge")
                fault = measure("fault")
                require(6.4 <= fault["actual_rate"] <= 9.6 and fault["completed"] == fault["actual_started"]
                        and .2 <= fault["success_rate"] <= .8, "partial selector failure invalid")
            elif case == "retry":
                burst("trigger")
                time.sleep(20)
                require(retry.collapsed(measure("fault", 25)), "retry collapse invalid")

            def record_call(event):
                with (folder / "public-tools.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")

            save(folder / "public-task.json", pilot.public_task())
            public = tools.PublicTools(backend, ledger.public_check, record_call, time.monotonic(),
                                       max_calls=64, profile="pilot", deadline_s=300)
            save(folder / "opened.json", {"opened_s": public.started, "host_unix_s": time.time(), "action_limit_s": 300})
            actor_error = None
            try:
                if actor == "runbook":
                    def send(call):
                        return json.loads(json.dumps(public.invoke(json.loads(json.dumps(call)))))

                    runbook.investigate(send)
                else:
                    public.invoke({"tool": "finish", "arguments": {"summary": "done"}})
            except Exception as exc:
                actor_error = f"{type(exc).__name__}: {exc}"
            save(folder / "completion.json", {"opened_s": public.started, "finished_s": public.finished_at,
                 "action_limit_s": public.deadline_s, "clock": public.clock(), "actor_error": actor_error})
            time.sleep(20)
            recovered = retry.healthy(measure("after-finish"))
            replayed = None
            if case == "retry" and recovered:
                burst("replay-trigger")
                time.sleep(20)
                replayed = retry.healthy(measure("after-replay"))
            after = json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))
            after_services = json.loads(runtime.kube(["-n", namespace, "get", "services", "-o", "json"]))
            save(folder / "after-deployments.json", after)
            save(folder / "after-services.json", after_services)
            old_specs = {obj["metadata"]["name"]: normalize_spec(obj) for obj in before["items"]}
            new_specs = {obj["metadata"]["name"]: normalize_spec(obj) for obj in after["items"]}
            if case == "selector":
                for name in ("frontend", "search"):
                    require(new_specs[name]["template"]["metadata"]["labels"].pop("service-route", None) == "frontend", "unexpected injector label")
            unchanged = old_specs == new_specs and len(new_specs) == 17
            old_services = {obj["metadata"]["name"]: copy.deepcopy(obj["spec"]) for obj in before_services["items"]}
            new_services = {obj["metadata"]["name"]: copy.deepcopy(obj["spec"]) for obj in after_services["items"]}
            for name in ("frontend", "search", "rate"):
                old_services[name].pop("selector", None)
                new_services[name].pop("selector", None)
            unchanged &= old_services == new_services
            ready = all(obj["status"].get("replicas") == obj["status"].get("updatedReplicas") == obj["status"].get("readyReplicas") == 1
                        and obj["status"].get("observedGeneration", 0) >= obj["metadata"]["generation"] for obj in after["items"])
            topology_ok = route("final", ["frontend"], wait=False) and ready and unchanged
            ledger.probe("final")
            metrics = workload.metrics.snapshot()
            save(folder / "final-metrics.json", metrics)
            stop_workload(workload)
            evaluation = ledger.evaluate(catalog)
            events = recorder.snapshot()
            post_done = sum(e["kind"] == "search" and public.finished_at is not None and e["started_s"] > public.finished_at for e in events)
            grade = pilot.grade(case, public, evaluation, recovered, replayed, topology_ok, retry.safe_policy(metrics), post_done)
            save(folder / "private-grade.json", grade)
            save(folder / "http-events.json", events)
            with workload._lock:
                driver = {"submissions": list(workload._submissions), "outcomes": list(workload._events)}
            save(folder / "driver.json", driver)
            require(len(driver["submissions"]) == len(driver["outcomes"]), "driver lost outcomes")
            (folder / "relay.jsonl").write_text(runtime.kube(["-n", private_namespace, "logs", "deployment/client"]) + "\n")
            resources(folder / "resources.json")
            entry.update({"grade": grade, "actor_error": actor_error,
                          "passed": actor_error is None and grade["success"] == (actor == "runbook")
                          and evaluation["invariants_preserved"] and unchanged and post_done >= 2})
            save(output / "summary.json", summary)
            print(json.dumps({"episode": index, **entry}), flush=True)
            require(entry["passed"], "fresh episode control failed")
            prior_ids.update(row["_id"] for row in ledger.snapshot()["reservations"])
            runtime.kube(["delete", "namespace", namespace, private_namespace, "--wait=true", "--timeout=60s"], timeout=75)
            remaining = json.loads(runtime.kube(["get", "namespaces", "-o", "json"]))
            save(folder / "namespaces-after.json", remaining)
            entry["reset_ok"] = not {namespace, private_namespace}.intersection(obj["metadata"]["name"] for obj in remaining["items"])
            require(entry["reset_ok"], "episode reset failed")
            workload = None
        summary["status"] = "passed"
    except BaseException as exc:
        summary.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        signal.alarm(0)
        if workload is not None:
            try:
                stop_workload(workload)
                if recorder is not None:
                    save(folder / "partial-http-events.json", recorder.snapshot())
                (folder / "partial-relay.jsonl").write_text(runtime.kube(["-n", private_namespace, "logs", "deployment/client"], check=False) + "\n")
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
        print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}), flush=True)


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
    signal.alarm(1800)
    run(args)


if __name__ == "__main__":
    main()
