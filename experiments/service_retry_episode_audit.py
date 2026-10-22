import argparse
import copy
import hashlib
import json
import os
import signal
import time
import uuid
from pathlib import Path

from service_operator import hotel, http_probe, kubernetes, retry, retry_episode, tools, workflow
from service_operator.kubernetes import LocalCluster, save
from service_retry_audit import INPUTS, KUBECTL_SHA256, guarded_workload, stop_workload


CONDITIONS = ("healthy", "repair", "restart_only", "premature_finish")
TASK = "Investigate this hotel-search alert. Keep existing bookings intact, verify search and booking, and decide whether changes are needed."


def public_script(public, condition):
    def call(tool, **args):
        reply = public.invoke({"tool": tool, "arguments": args})
        if not reply["ok"]:
            raise RuntimeError("scripted public call failed: " + tool)
        return reply["result"]

    def wait_rollouts(names):
        for _ in range(12):
            ready = True
            for name in names:
                obj = call("inspect", resource="deployment", name=name)
                state = obj["status"]
                ready &= (state.get("readyReplicas") == state.get("updatedReplicas") == state.get("replicas") == 1
                          and state.get("observedGeneration", 0) >= obj["generation"])
            if ready:
                return
            call("wait", seconds=2)
        raise RuntimeError("public rollout observations did not converge")

    if condition != "premature_finish":
        call("clock")
        call("inspect", resource="deployment", name="search")
        call("metrics", service="search")
        call("metrics", service="rate")
    if condition == "healthy":
        call("check")
    elif condition == "repair":
        call("logs", deployment="search", lines=10)
        call("check")
        call("configure_rate", qps_limit=50, queue_capacity=256)
        call("configure_search", timeout_ms=1000, max_attempts=2, backoff_ms=50, backoff_multiplier=2.0, jitter=0.2)
        wait_rollouts(["rate", "search"])
        call("wait", seconds=10)
        call("wait", seconds=10)
        call("metrics", service="rate")
        call("check")
    elif condition == "restart_only":
        call("restart", deployment="frontend")
        wait_rollouts(["frontend"])
    call("finish", summary="investigation complete")


def normalize_spec(obj):
    result = copy.deepcopy(obj["spec"])
    name = obj["metadata"]["name"]
    template = result["template"]
    if name in ("frontend", "search", "rate"):
        annotations = template["metadata"].get("annotations", {})
        annotations.pop("kubectl.kubernetes.io/restartedAt", None)
        if not annotations:
            template["metadata"].pop("annotations", None)
    for container in template["spec"]["containers"]:
        fields = tools.POLICY_FIELDS.get("configure_" + name, {})
        allowed = {spec[0] for spec in fields.values()}
        container["env"] = sorted([row for row in container.get("env", []) if row["name"] not in allowed], key=lambda row: row["name"])
    return result


def audit_permissions(runtime, backend, private_namespace, output):
    cases = [("get", "services", backend.namespace, None, True),
             ("patch", "services", backend.namespace, None, False),
             *[("patch", "deployments.apps", backend.namespace, name, True) for name in ("frontend", "search", "rate")],
             ("patch", "deployments.apps", backend.namespace, "mongodb-reservation", False),
             *[("get", "services/proxy", backend.namespace, name, True) for name in ("search-metrics", "rate-metrics")],
             ("get", "services/proxy", backend.namespace, "frontend", False),
             ("list", "pods", private_namespace, None, False),
             ("get", "secrets", backend.namespace, None, False),
             ("create", "pods/exec", backend.namespace, None, False),
             ("patch", "rolebindings.rbac.authorization.k8s.io", backend.namespace, None, False),
             ("delete", "deployments.apps", backend.namespace, None, False),
             ("get", "nodes", None, None, False)]
    checks = []
    for verb, resource, namespace, name, expected in cases:
        base, _, subresource = resource.partition("/")
        target = base + "/" + name if name else base
        args = [*backend.identity, "auth", "can-i", verb, target]
        if subresource:
            args += ["--subresource=" + subresource]
        if namespace:
            args += ["-n", namespace]
        answer = runtime.kube(args, check=False)
        checks.append({"verb": verb, "resource": resource, "namespace": namespace, "name": name,
                       "answer": answer, "expected": expected, "passed": answer == ("yes" if expected else "no")})
    for label, args in [("private pods", [*backend.identity, "-n", private_namespace, "get", "pods", "-o", "json"]),
                        ("frontend proxy", [*backend.identity, "get", "--raw", f"/api/v1/namespaces/{backend.namespace}/services/frontend/proxy/hotels"])]:
        runtime.kube(args, check=False)
        metadata = json.loads((runtime.output / f"command-{runtime.count:04d}.json").read_text())
        raw = (runtime.output / f"command-{runtime.count:04d}.log").read_text()
        checks.append({"operation": label, "passed": metadata["returncode"] == 1 and "Forbidden" in raw})
    save(output, checks)
    if not all(row["passed"] for row in checks):
        raise RuntimeError("native permissions differ from the retry contract")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = LocalCluster(output, args.kind)
    started = time.monotonic()
    result = {"status": "running", "run_id": runtime.name, "episodes": [], "cleanup_ok": False}
    images = retry.images(args.availability_build)
    kubectl = args.inputs.resolve() / "bin/kubectl"
    if hashlib.sha256(kubectl.read_bytes()).hexdigest() != KUBECTL_SHA256:
        raise ValueError("kubectl differs from the pinned binary")
    init_scripts = {name: (args.inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh").read_text() for name in ("geo", "rate")}
    for name, text in init_scripts.items():
        body = text.encode()
        relative = f"applications/hotelReservation/k8s-{name}-mongo.sh"
        if hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() != INPUTS[relative]:
            raise ValueError("Mongo initialization source differs")
    original_env = {key: os.environ.get(key) for key in ("PATH", "KUBECONFIG")}
    source_dir = output / "source"
    source_dir.mkdir()
    sources = [Path(__file__), Path(__file__).with_name("service_retry_audit.py"), args.workload, args.availability_build,
               Path("patches/hotel-reservation-availability.patch"),
               *[args.inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh" for name in ("geo", "rate")],
               *[Path(module.__file__) for module in
               (hotel, http_probe, kubernetes, retry, retry_episode, tools, workflow)]]
    hashes = {}
    for path in sources:
        (source_dir / path.name).write_bytes(path.read_bytes())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "protocol.txt").write_bytes(Path("docs/SERVICE_RETRY_EPISODE_PROTOCOL.md").read_bytes())
    save(output / "protocol.json", {"version": 1, "scope": "scripted retry JSON boundary; no model, training or official score",
         "conditions": CONDITIONS, "source_sha256": hashes, "images": images,
         "node_image": kubernetes.NODE_IMAGE, "kind_sha256": kubernetes.KIND_SHA256,
         "kubectl_sha256": KUBECTL_SHA256,
         "action_limit_s": 300, "max_tool_calls": 64, "max_checks": 2, "overall_limit_s": 1800,
         "traffic": {"base_rps": 8, "burst_rps": 40, "burst_s": 10, "deadline_s": 4, "workers": 192},
         "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048}})
    workload = None

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def resources(path):
        value = {name: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + name])
                 for name in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
        save(path, value)
        require(dict(line.split() for line in value["memory.events"].splitlines())["oom_kill"] == "0", "node OOM kill")

    try:
        original = runtime.command(runtime.docker + ["ps", "--format", "json"])
        save(output / "unrelated-before.json", [json.loads(line) for line in original.splitlines()])
        runtime.start()
        runtime.load_images(images)
        os.environ["PATH"] = str(kubectl.parent) + os.pathsep + (original_env["PATH"] or "")
        os.environ["KUBECONFIG"] = str(runtime.kubeconfig)
        for index, condition in enumerate(CONDITIONS, 1):
            folder = output / f"episode-{index}"
            folder.mkdir()
            namespace, private_namespace = "app-" + uuid.uuid4().hex[:10], "observer-" + uuid.uuid4().hex[:10]
            entry = {"condition": condition, "namespace": namespace, "private_namespace": private_namespace}
            result["episodes"].append(entry)
            save(output / "summary.json", result)
            print("episode " + str(index) + ": " + condition, flush=True)
            runtime.apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}} for name in (namespace, private_namespace)])
            objects = retry.profile(namespace, runtime.aliases, init_scripts)
            save(folder / "application.json", objects)
            runtime.apply([*objects, *retry_episode.metric_services(namespace), *tools.operator_role(namespace, "retry")])
            deadline = time.monotonic() + 180
            for obj in objects:
                if obj["kind"] == "Deployment":
                    remaining = max(1, int(deadline - time.monotonic()))
                    require(time.monotonic() < deadline, "rollout deadline")
                    runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + obj["metadata"]["name"], f"--timeout={remaining}s"], timeout=remaining + 10)
            actual_binary = runtime.kube(["-n", namespace, "exec", "deployment/reservation", "--", "sha256sum", "/go/bin/reservation"])
            save(folder / "reservation-binary.json", {"sha256sum": actual_binary})
            require(actual_binary.split()[0] == images["app"]["binary_sha256"], "runtime binary differs")
            rows = json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-profile", "--", "mongo", "--quiet", "--eval", retry.CATALOG_JS]))
            catalog = {row["id"]: row for row in rows}
            save(folder / "catalog.json", rows)
            require(len(catalog) == 80, "catalog is incomplete")
            recorder = retry.RecordedRequests(folder / "http.jsonl")
            workload_class = retry.load_workload(args.workload, recorder)
            workload = guarded_workload(workload_class, namespace)
            workload.frontend.start()
            ledger = retry_episode.SearchLedger(runtime, namespace, workload, folder / "private-ledger")
            require(hotel.fresh_seed(ledger.before), "reset seed differs")
            warm = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                response = http_probe.request(f"http://127.0.0.1:{workload.frontend.local_port}" + workload._PATH, 4)
                warm.append(response)
                save(folder / "warmup.json", warm)
                try:
                    retry.select_hotel(response, catalog)
                    break
                except ValueError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("search warmup deadline")
            ledger.probe("baseline")
            require(ledger.evaluate(catalog, "baseline")["final_valid"], "baseline workflow invalid")
            baseline_state = ledger.snapshot()
            backend = retry_episode.RetryBackend(runtime, namespace)
            audit_permissions(runtime, backend, private_namespace, folder / "permissions.json")
            for service in ("search", "rate"):
                save(folder / (service + "-public-metrics.json"), backend.perform("metrics", {"service": service}))

            def deployments():
                return json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))

            before_deployments = deployments()
            save(folder / "before-deployments.json", before_deployments)

            def measure(name, seconds):
                before = workload.metrics.snapshot()
                t0 = time.monotonic()
                time.sleep(seconds)
                t1 = time.monotonic()
                after = workload.metrics.snapshot()
                time.sleep(4.25)
                sample = retry.window(recorder.snapshot(), t0, t1, catalog, before, after)
                record = {"start_s": t0, "end_s": t1, "before": before, "after": after, "sample": sample}
                save(folder / (name + ".json"), record)
                return record

            def burst(name):
                t0 = time.monotonic()
                workload.run_trigger(40, 10)
                t1 = time.monotonic()
                time.sleep(4.25)
                cohort = [row for row in recorder.snapshot() if row["kind"] == "search" and t0 <= row["started_s"] < t1]
                record = {"start_s": t0, "end_s": t1, "actual_started": len(cohort), "actual_rate": len(cohort) / (t1 - t0),
                          "completed": sum("finished_s" in row for row in cohort)}
                save(folder / (name + ".json"), record)
                require(record["actual_rate"] >= 32 and record["completed"] == record["actual_started"], "burst delivery invalid")

            workload.start()
            time.sleep(20)
            baseline = measure("healthy", 20)
            require(retry.healthy(baseline["sample"], initial=True), "healthy protected load invalid")
            require(ledger.snapshot() == baseline_state, "baseline search changed data")
            if condition != "healthy":
                burst("trigger")
                time.sleep(20)
                fault = measure("fault", 25)
                require(retry.collapsed(fault["sample"]), "fault did not sustain")
            opened = time.monotonic()
            def record_call(event):
                with (folder / "public-tools.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")

            public = tools.PublicTools(backend, ledger.public_check, record_call, opened,
                                       max_calls=64, profile="retry", deadline_s=300)
            save(folder / "public-task.json", {"task": TASK, "tools": tools.tool_definitions("retry"),
                 "clock": public.clock(), "check_budget": 2, "completion": "declares completion; workload and private evaluation continue",
                 "policy_updates": "request a rollout; inspect generation/readiness and runtime metrics before assuming application"})
            actor_error = None
            try:
                public_script(public, condition)
            except Exception as exc:
                actor_error = f"{type(exc).__name__}: {exc}"
            save(folder / "completion.json", {"opened_s": opened, "finished_s": public.finished_at,
                 "action_limit_s": 300, "clock": public.clock(), "actor_error": actor_error})
            time.sleep(20)
            recovered = measure("after-finish", 20)
            recovered_ok = retry.healthy(recovered["sample"])
            replayed = None
            if condition == "repair" and recovered_ok:
                burst("replay-trigger")
                time.sleep(20)
                replayed = retry.healthy(measure("after-replay", 20)["sample"])
            after_deployments = deployments()
            slices = json.loads(runtime.kube(["-n", namespace, "get", "endpointslices", "-o", "json"]))
            save(folder / "after-deployments.json", after_deployments)
            save(folder / "endpoints.json", slices)
            before = {obj["metadata"]["name"]: normalize_spec(obj) for obj in before_deployments["items"]}
            after = {obj["metadata"]["name"]: normalize_spec(obj) for obj in after_deployments["items"]}
            require(before == after and len(after) == 17, "unrelated deployment fields changed")
            topology_ok = all(obj["status"].get("readyReplicas") == obj["status"].get("updatedReplicas") == obj["status"].get("replicas") == 1
                and obj["status"].get("observedGeneration", 0) >= obj["metadata"]["generation"] for obj in after_deployments["items"])
            topology_ok &= all(any(obj["metadata"]["labels"].get("kubernetes.io/service-name") == name
                and any(endpoint.get("conditions", {}).get("ready") for endpoint in obj.get("endpoints", [])) for obj in slices["items"])
                for name in ("frontend", "search", "rate"))
            ledger.probe("final")
            stop_workload(workload)
            evaluation = ledger.evaluate(catalog)
            metrics = workload.metrics.snapshot()
            save(folder / "final-metrics.json", metrics)
            post_done = sum(event["kind"] == "search" and public.finished_at is not None and event["started_s"] > public.finished_at for event in recorder.snapshot())
            grade = retry_episode.grade(public, evaluation, recovered_ok, replayed, topology_ok, retry.safe_policy(metrics), condition == "healthy", post_done)
            save(folder / "private-grade.json", grade)
            save(folder / "http-events.json", recorder.snapshot())
            with workload._lock:
                driver = {"submissions": list(workload._submissions), "outcomes": list(workload._events)}
                save(folder / "driver.json", driver)
            stop_workload(workload)
            require(len(driver["submissions"]) == len(driver["outcomes"]), "workload offers lost outcomes")
            resources(folder / "resources.json")
            entry.update({"grade": grade, "passed": actor_error is None and grade["success"] == (condition in ("healthy", "repair"))
                          and evaluation["invariants_preserved"] and grade["post_completion_searches"] >= 2})
            save(output / "summary.json", result)
            print(json.dumps({"condition": condition, "grade": grade, "passed": entry["passed"]}), flush=True)
            require(entry["passed"], "public episode control failed")
            runtime.kube(["delete", "namespace", namespace, private_namespace, "--wait=true", "--timeout=60s"], timeout=75)
            remaining = json.loads(runtime.kube(["get", "namespaces", "-o", "json"]))
            entry["reset_ok"] = not {namespace, private_namespace}.intersection(obj["metadata"]["name"] for obj in remaining["items"])
            require(entry["reset_ok"], "episode namespaces survived reset")
            workload = None
        result["status"] = "passed"
    except BaseException as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        signal.alarm(0)
        if workload is not None:
            try:
                stop_workload(workload)
            except Exception as exc:
                result["driver_cleanup_error"] = str(exc)
        result["cleanup_ok"] = runtime.close()["ok"]
        if not result["cleanup_ok"]:
            result["status"] = "failed"
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        result["elapsed_s"] = time.monotonic() - started
        save(output / "summary.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=Path, default=Path("results/service-kube-runtime-20261021/kind"))
    parser.add_argument("--inputs", type=Path, default=Path("results/service-retry-source-20261021"))
    parser.add_argument("--workload", type=Path, default=Path("results/service-plan-source-20261019/sregym/sregym/generators/workload/hotel_search.py"))
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
