import argparse
import hashlib
import json
import shlex
import signal
import time
import types
import urllib.parse
from pathlib import Path

from service_operator import hotel, http_probe, kubernetes, selector, workflow
from service_operator.kubernetes import LocalCluster, save


PHASES = (("healthy", 4), ("fault", 16), ("restart_only", 16),
          ("repaired", 4), ("false_response", 4), ("restored", 4))


class InjectorAdapter:
    def __init__(self, runtime, namespace, output):
        self.runtime, self.namespace, self.output = runtime, namespace, output

    def record(self, entry):
        with self.output.open("a") as handle:
            handle.write(json.dumps({"unix_s": time.time(), **entry}) + "\n")

    def patch_deployment(self, name, namespace, patch):
        if namespace != self.namespace or name not in ("frontend", "search"):
            raise ValueError("injector escaped its declared targets")
        self.record({"operation": "patch_deployment", "deployment": name, "patch": patch})
        return self.runtime.kube(["-n", namespace, "patch", "deployment", name,
                                  "--type=merge", "-p", json.dumps(patch)])

    def exec_command(self, command):
        args = shlex.split(command)
        if args[0] != "kubectl" or "-n" not in args or args[args.index("-n") + 1] != self.namespace:
            raise ValueError("unexpected upstream command")
        self.record({"operation": "kubectl", "args": args[1:]})
        return self.runtime.kube(args[1:], timeout=130)


def run(output, kind, injector_path, context):
    output.mkdir(parents=True, exist_ok=False)
    runtime = LocalCluster(output, kind, context)
    started = time.monotonic()
    result = {"status": "running", "run_id": runtime.name, "cycles": [], "cleanup_ok": False}
    functions = selector.load_selector_functions(injector_path)
    sources = [Path(__file__), injector_path, *[Path(module.__file__)
               for module in (hotel, http_probe, kubernetes, selector, workflow)]]
    source_hashes = {}
    source_dir = output / "source"
    source_dir.mkdir()
    for source in sources:
        body = source.read_bytes()
        (source_dir / source.name).write_bytes(body)
        source_hashes[source.name] = hashlib.sha256(body).hexdigest()
    images = {"client": hotel.CLIENT_IMAGE, **hotel.IMAGES}
    save(output / "protocol.json", {
        "protocol_version": 2,
        "scope": "local reduced Kubernetes profile; scripted controls, no official oracle/benchmark or agent score",
        "upstream_revision": "482846e48c6cd7ae1a259d8412ae4ebc76a035bd",
        "upstream_application_revision": "2b2f9c6c2e97c44abbfcc44af1cf2f994bbb04f8",
        "images": images, "node_image": kubernetes.NODE_IMAGE, "kind_sha256": kubernetes.KIND_SHA256,
        "source_sha256": source_hashes, "phases": PHASES, "cycles": 2,
        "request_deadline_s": 1, "settle_s": 0.25, "endpoint_deadline_s": 30,
        "frontend_readiness_deadline_s": 20, "overall_deadline_s": 1200,
        "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048},
        "profile": "booking chain plus real search, in-cluster client, supplied false-response control",
        "traffic": "sequential unique one-night writes over frontend ClusterIP; new connection, no retry",
        "reset": "delete entire owned application namespace; next cycle uses fresh emptyDir databases",
        "advance": "healthy/repair/restored all confirmed; fault/restart mixed success/failure with both endpoints; fake acknowledgements rejected; prior rows/capacity/reset/cleanup valid",
        "limitations": "no concurrent workload, full observability stack, native performance, auth/durability guarantee or learning result",
    })
    remote_code = Path(http_probe.__file__).read_text()

    def http(namespace, url, deadline=1):
        raw = runtime.kube(["-n", namespace, "exec", "deployment/client", "--", "python", "-c",
                            remote_code, url, str(deadline)], timeout=15)
        return json.loads(raw)

    def snapshot(namespace):
        return json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-reservation", "--",
                                        "mongo", "--quiet", "--eval", hotel.SNAPSHOT_JS]))

    def route_state(namespace):
        state = {}
        for key, resource in (("service", "service/frontend"), ("deployments", "deployments"),
                              ("pods", "pods"), ("replicasets", "replicasets"), ("slices", "endpointslices")):
            args = ["-n", namespace, "get", resource, "-o", "json"]
            if key == "slices":
                args += ["-l", "kubernetes.io/service-name=frontend"]
            state[key] = json.loads(runtime.kube(args))
        return state

    def wait_route(namespace, expected, path):
        deadline = time.monotonic() + 30
        observations = []
        while time.monotonic() < deadline:
            state = route_state(namespace)
            observations.append(state)
            save(path, observations)
            if selector.check_route(state, expected):
                return state
            time.sleep(1)
        raise RuntimeError("endpoint identity did not converge")

    def warm_frontend(namespace, path):
        observations = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            pods = json.loads(runtime.kube(["-n", namespace, "get", "pods", "-l", "io.kompose.service=frontend", "-o", "json"]))
            for pod in pods["items"]:
                if not pod["metadata"].get("deletionTimestamp") and pod.get("status", {}).get("podIP"):
                    url = "http://" + pod["status"]["podIP"] + ":5000/user?username=Cornell_1&password=1111111111"
                    response = http(namespace, url)
                    observations.append({"pod": pod["metadata"]["name"], "response": response})
                    save(path, observations)
                    if response.get("body") == {"message": "Login successfully!"}:
                        return
            time.sleep(1)
        raise RuntimeError("real frontend login did not become ready")

    def patch_selector(namespace, labels):
        return runtime.kube(["-n", namespace, "patch", "service", "frontend", "--type=json", "-p",
                             json.dumps([{"op": "replace", "path": "/spec/selector", "value": labels}])])

    try:
        print("starting isolated Kubernetes node", flush=True)
        runtime.start()
        runtime.load_images(images)
        for cycle in (1, 2):
            folder = output / f"cycle-{cycle}"
            folder.mkdir()
            namespace = f"hotel-cycle-{cycle}"
            entry = {"cycle": cycle, "namespace": namespace, "phases": []}
            result["cycles"].append(entry)
            runtime.apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}}])
            objects = hotel.kubernetes_profile(namespace, runtime.aliases)
            save(folder / "application.json", objects)
            apps = {"frontend", "search", "reservation", "user"}
            support = [obj for obj in objects if obj["kind"] != "Deployment" or obj["metadata"]["name"] not in apps]
            runtime.apply(support)
            for obj in support:
                if obj["kind"] == "Deployment":
                    runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + obj["metadata"]["name"], "--timeout=90s"], timeout=100)
            runtime.apply([obj for obj in objects if obj["kind"] == "Deployment" and obj["metadata"]["name"] in apps])
            for name in sorted(apps):
                runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + name, "--timeout=90s"], timeout=100)
            warm_frontend(namespace, folder / "initial-readiness.json")
            initial = snapshot(namespace)
            save(folder / "initial.json", initial)
            entry["fresh_seed"] = hotel.fresh_seed(initial)
            if not entry["fresh_seed"]:
                raise RuntimeError("namespace reset has invalid seed state")
            before = initial
            adapter = InjectorAdapter(runtime, namespace, folder / "injector-actions.jsonl")
            target = types.SimpleNamespace(kubectl=adapter, namespace=namespace)
            for phase, count in PHASES:
                print(f"cycle {cycle}: {phase}", flush=True)
                phase_dir = folder / phase
                phase_dir.mkdir()
                phase_entry = {"name": phase, "started_s": time.monotonic() - started, "requests": []}
                entry["phases"].append(phase_entry)
                save(output / "summary.json", result)
                if phase == "fault":
                    functions["inject_service_wrong_pod_selection"](target, ["frontend", "search"])
                elif phase == "restart_only":
                    runtime.kube(["-n", namespace, "rollout", "restart", "deployment/frontend"])
                    runtime.kube(["-n", namespace, "rollout", "status", "deployment/frontend", "--timeout=120s"], timeout=130)
                elif phase == "repaired":
                    functions["recover_service_wrong_pod_selection"](target, ["frontend", "search"])
                elif phase in ("false_response", "restored"):
                    patch_selector(namespace, {"io.kompose.service": "false-response" if phase == "false_response" else "frontend"})
                expected_routes = ["frontend", "search"] if phase in ("fault", "restart_only") else [
                    "false-response" if phase == "false_response" else "frontend"]
                state = wait_route(namespace, expected_routes, phase_dir / "routing.json")
                warm_frontend(namespace, phase_dir / "readiness.json")
                if snapshot(namespace) != before:
                    raise RuntimeError("application state changed during transition")
                phase_entry["measurement_started_s"] = time.monotonic() - started
                for number in range(count):
                    expected = {"hotelId": "1", "customerName": f"{runtime.name}-{cycle}-{phase}-{number}",
                                "inDate": "2026-11-01", "outDate": "2026-11-02", "number": 1}
                    query = urllib.parse.urlencode({**expected, "username": "Cornell_1", "password": "1111111111"})
                    request_path = phase_dir / f"request-{number:02d}.json"
                    event = {"number": number, "expected": expected, "before": before,
                             "started_unix_s": time.time(), "url": "http://frontend:5000/reservation?" + query}
                    save(request_path, event)
                    response = http(namespace, event["url"])
                    event["response"] = response
                    save(request_path, event)
                    immediate = snapshot(namespace)
                    event["immediate"] = immediate
                    save(request_path, event)
                    time.sleep(0.25)
                    after = snapshot(namespace)
                    verdict = workflow.check_booking(before["reservations"], after["reservations"], expected, response)
                    verdict["capacity_preserved"] = before["capacity"] == after["capacity"]
                    verdict["stable_after_response"] = immediate == after
                    verdict["accepted"] &= verdict["capacity_preserved"] and verdict["stable_after_response"]
                    event.update({"after": after, "verdict": verdict, "finished_unix_s": time.time()})
                    save(request_path, event)
                    phase_entry["requests"].append({"number": number, "response": response, "verdict": verdict})
                    save(output / "summary.json", result)
                    if (not verdict["prior_preserved"] or not verdict["capacity_preserved"]
                            or not verdict["stable_after_response"]
                            or verdict["inserted_count"] and not verdict["exact_new_write"]):
                        raise RuntimeError("invalid or unstable database write; not retried")
                    before = after
                accepted = sum(item["verdict"]["accepted"] for item in phase_entry["requests"])
                phase_entry["accepted"] = accepted
                phase_entry["http_successes"] = sum(item["response"]["status"] == 200 for item in phase_entry["requests"])
                phase_entry["finished_s"] = time.monotonic() - started
                if phase in ("fault", "restart_only"):
                    passed = 0 < accepted < count
                elif phase == "false_response":
                    passed = accepted == 0 and all(item["verdict"]["acknowledged"] and item["verdict"]["inserted_count"] == 0
                                                   for item in phase_entry["requests"])
                else:
                    passed = accepted == count
                phase_entry["passed"] = passed
                print(f"{phase}: {accepted}/{count} valid bookings", flush=True)
                if not passed:
                    raise RuntimeError("phase failed its frozen control rule")
                wait_route(namespace, expected_routes, phase_dir / "routing-after.json")
                save(output / "summary.json", result)
            save(folder / "final.json", snapshot(namespace))
            save(folder / "events.json", json.loads(runtime.kube(["-n", namespace, "get", "events", "-o", "json"])))
            runtime.kube(["delete", "namespace", namespace, "--wait=true", "--timeout=60s"], timeout=70)
            remaining = json.loads(runtime.kube(["get", "namespaces", "-o", "json"]))
            entry["reset_ok"] = namespace not in [item["metadata"]["name"] for item in remaining["items"]]
            if not entry["reset_ok"]:
                raise RuntimeError("owned namespace remained after reset")
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        cleanup = runtime.close()
        result["cleanup_ok"] = cleanup["ok"]
        if not cleanup["ok"]:
            result["status"] = "failed"
        result["elapsed_s"] = time.monotonic() - started
        save(output / "summary.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "cycles"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=Path, default=Path("results/service-kube-runtime-20261021/kind"))
    parser.add_argument("--injector", type=Path, default=Path("results/service-kube-runtime-20261021/inject_virtual.py"))
    parser.add_argument("--context", default="desktop-linux")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(1200)
    run(args.output.resolve(), args.kind, args.injector, args.context)


if __name__ == "__main__":
    main()
