import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from service_operator import hotel, http_probe, kubernetes, retry, workflow
from service_operator.kubernetes import LocalCluster, save


KUBECTL_SHA256 = "d9e0946a4446cfd66495676273aa71c64566a49e63580136eb138ef6b562c4d9"
INPUTS = {
    "sregym/conductor/oracles/search_rate_retry_mitigation.py": "42cfcf31f57436f78e040e86c0282e35df0d85fa",
    "applications/hotelReservation/k8s-geo-mongo.sh": "51062215e52aca767f85d489c9109c38fe52c190",
    "applications/hotelReservation/k8s-rate-mongo.sh": "67caba4636a9476643045c368760d0565be33ebd",
}


def guarded_workload(workload_class, namespace):
    workload = workload_class(namespace)
    start = workload.frontend.start
    lock = threading.Lock()

    def start_unless_stopping(*args, **kwargs):
        with lock:
            if workload._stop.is_set():
                raise ValueError("workload is stopping")
            return start(*args, **kwargs)

    workload.frontend.start = start_unless_stopping
    return workload


def stop_workload(workload):
    workload._stop.set()
    if workload._thread is not None:
        workload._thread.join(timeout=3)
    if workload._executor is not None:
        workload._executor.shutdown(wait=True, cancel_futures=True)
        workload._executor = None
    workload.stop()


def stop_stats(process):
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime = LocalCluster(output, args.kind, args.context)
    recorder = retry.RecordedRequests(output / "http.jsonl")
    source_dir = output / "source"
    source_dir.mkdir()
    hashes = {}
    for source in [Path(__file__), args.workload, *[Path(module.__file__) for module in
                   (hotel, http_probe, kubernetes, retry, workflow)]]:
        body = source.read_bytes()
        (source_dir / source.name).write_bytes(body)
        hashes[source.name] = hashlib.sha256(body).hexdigest()
    for relative, expected in INPUTS.items():
        body = (args.inputs / relative).read_bytes()
        if hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() != expected:
            raise ValueError("pinned input differs: " + relative)
        (source_dir / Path(relative).name).write_bytes(body)
        hashes[relative] = hashlib.sha256(body).hexdigest()
    kubectl = args.inputs.resolve() / "bin/kubectl"
    if hashlib.sha256(kubectl.read_bytes()).hexdigest() != KUBECTL_SHA256:
        raise ValueError("kubectl binary differs from the pinned release")
    workload_class = retry.load_workload(args.workload, recorder)
    init_scripts = {name: (args.inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh").read_text()
                    for name in ("geo", "rate")}
    (output / "protocol.txt").write_bytes(Path("docs/SERVICE_RETRY_PROTOCOL.md").read_bytes())
    save(output / "protocol.json", {
        "version": 1, "scope": "scripted local fit/mechanism control; not the official oracle or an agent score",
        "main_revision": "482846e48c6cd7ae1a259d8412ae4ebc76a035bd",
        "application_revision": "2b2f9c6c2e97c44abbfcc44af1cf2f994bbb04f8",
        "images": hotel.IMAGES, "node_image": kubernetes.NODE_IMAGE, "kind_sha256": kubernetes.KIND_SHA256,
        "kubectl_sha256": KUBECTL_SHA256, "requests_version": requests.__version__, "source_sha256": hashes,
        "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048}, "overall_deadline_s": 1500,
        "traffic": {"base_rps": 8, "burst_rps": 40, "burst_s": 10, "workers": 192, "deadline_s": 4},
        "vulnerable": retry.VULNERABLE, "mitigated": retry.MITIGATED,
        "phases": ["healthy", "burst", "sustained", "frontend_restart", "repair", "replay", "namespace_reset"],
        "protected_load": "8 rps remains active during interventions, window drains and state checks",
        "transport": "audited upstream localhost kubectl port-forward; no global kubeconfig/PATH changes",
        "reset": "delete owned namespace; recreate 17 deployments with fresh emptyDir state",
    })
    result = {"status": "running", "run_id": runtime.name, "phases": [], "cleanup_ok": False}
    workload = None
    stats = None
    stats_log = None
    old_env = {key: os.environ.get(key) for key in ("PATH", "KUBECONFIG")}
    namespace = "retry-app"
    current_state = None
    catalog = None
    baseline_names = None

    def phase(name):
        entry = {"name": name, "started_s": time.monotonic() - started}
        result["phases"].append(entry)
        save(output / "summary.json", result)
        print(name, flush=True)
        return entry

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def snapshot():
        return json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-reservation", "--",
                                       "mongo", "--quiet", "--eval", hotel.SNAPSHOT_JS]))

    def usage(label):
        data = {name: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + name])
                for name in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
        save(output / (label + "-resources.json"), data)
        counters = dict(line.split() for line in data["memory.events"].splitlines())
        require(int(counters["oom_kill"]) == 0, "node OOM kill invalidated resource fit")
        return data

    def topology(label):
        deployments = json.loads(runtime.kube(["-n", namespace, "get", "deployments", "-o", "json"]))
        slices = json.loads(runtime.kube(["-n", namespace, "get", "endpointslices", "-o", "json"]))
        pods = json.loads(runtime.kube(["-n", namespace, "get", "pods", "-o", "json"]))
        save(output / (label + "-topology.json"), {"deployments": deployments, "slices": slices, "pods": pods})
        names = {obj["metadata"]["name"] for obj in deployments["items"]}
        require(len(names) == 17 and (baseline_names is None or names == baseline_names), "deployment set changed")
        for obj in deployments["items"]:
            status = obj.get("status", {})
            require(obj["spec"]["replicas"] == 1 and status.get("readyReplicas") == 1
                    and status.get("updatedReplicas") == 1 and status.get("replicas") == 1
                    and status.get("observedGeneration", 0) >= obj["metadata"]["generation"], "deployment not current and ready")
        for name in ("frontend", "search", "rate"):
            require(any(obj["metadata"]["labels"].get("kubernetes.io/service-name") == name
                        and any(endpoint.get("conditions", {}).get("ready") for endpoint in obj.get("endpoints", []))
                        for obj in slices["items"]), "missing ready endpoint: " + name)
        require(all(pod.get("status", {}).get("phase") != "Failed" for pod in pods["items"]), "failed app pod")
        return names

    def install(label):
        runtime.kube(["create", "namespace", namespace])
        objects = retry.profile(namespace, runtime.aliases, init_scripts)
        save(output / (label + "-manifest.json"), objects)
        runtime.apply(objects)
        deadline = time.monotonic() + 180
        for obj in objects:
            if obj["kind"] == "Deployment":
                remaining = max(1, int(deadline - time.monotonic()))
                require(time.monotonic() < deadline, "application rollout deadline")
                runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + obj["metadata"]["name"],
                              f"--timeout={remaining}s"], timeout=remaining + 10)

    def warm(label):
        attempts = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            port = workload.frontend.start()
            response = http_probe.request(f"http://127.0.0.1:{port}" + workload._PATH, 4)
            attempts.append(response)
            save(output / (label + "-warmup.json"), attempts)
            try:
                retry.select_hotel(response, catalog)
                return
            except ValueError:
                time.sleep(0.5)
        raise RuntimeError("functional search warmup deadline")

    def save_driver(label):
        with workload._lock:
            save(output / (label + "-driver-events.json"),
                 {"submissions": list(workload._submissions), "completed": list(workload._events)})

    def booking(label):
        nonlocal current_state
        path = output / (label + "-booking.json")
        require(snapshot() == current_state, "state changed before booking")
        port = workload.frontend.start()
        search = http_probe.request(f"http://127.0.0.1:{port}" + workload._PATH, 4)
        event = {"search": search, "before": current_state, "started_unix_s": time.time()}
        save(path, event)
        selected = retry.select_hotel(search, catalog)
        expected = {"hotelId": selected, "customerName": runtime.name + "-" + label,
                    "inDate": "2015-04-09", "outDate": "2015-04-10", "number": 1}
        event["expected"] = expected
        save(path, event)
        url = f"http://127.0.0.1:{port}/reservation?" + urlencode({**expected, "username": "Cornell_1", "password": "1111111111"})
        event["response"] = http_probe.request(url, 4)
        save(path, event)
        event["immediate"] = snapshot()
        save(path, event)
        time.sleep(0.25)
        event["after"] = snapshot()
        event["verdict"] = workflow.check_booking(current_state["reservations"], event["after"]["reservations"], expected, event["response"])
        event["verdict"]["capacity_preserved"] = current_state["capacity"] == event["after"]["capacity"]
        event["verdict"]["stable"] = event["immediate"] == event["after"]
        save(path, event)
        require(all(event["verdict"][key] for key in ("accepted", "capacity_preserved", "stable")), "search-selected booking failed; not retried")
        current_state = event["after"]

    def measure(label, duration, check, initial=False):
        before = workload.metrics.snapshot()
        t0 = time.monotonic()
        time.sleep(duration)
        t1 = time.monotonic()
        after = workload.metrics.snapshot()
        time.sleep(4.25)
        sample = retry.window(recorder.snapshot(), t0, t1, catalog, before, after)
        record = {"start_s": t0, "end_s": t1, "before_metrics": before, "after_metrics": after, "sample": sample}
        record["passed"] = check(sample, initial=initial) if check is retry.healthy else check(sample)
        record["safe_policy"] = retry.safe_policy(after)
        save(output / (label + "-measurement.json"), record)
        print(json.dumps({"phase": label, **sample, "passed": record["passed"]}), flush=True)
        require(snapshot() == current_state, "search traffic changed stored reservations")
        usage(label)
        require(record["passed"] and record["safe_policy"], "measurement failed: " + label)
        return record

    def burst(label):
        t0 = time.monotonic()
        workload.run_trigger(40, 10)
        t1 = time.monotonic()
        time.sleep(4.25)
        events = [event for event in recorder.snapshot() if event["kind"] == "search" and t0 <= event["started_s"] < t1]
        record = {"start_s": t0, "end_s": t1, "duration_s": t1 - t0,
                  "actual_started": len(events), "actual_rate": len(events) / (t1 - t0),
                  "completed": sum("finished_s" in event for event in events)}
        save(output / (label + "-burst.json"), record)
        require(record["actual_rate"] >= 32 and record["completed"] == len(events), "burst not delivered by driver")
        return record

    try:
        containers = runtime.command(runtime.docker + ["ps", "--format", "json"])
        save(output / "unrelated-before.json", [json.loads(line) for line in containers.splitlines()])
        runtime.start()
        stats_log = (output / "docker-stats.jsonl").open("w")
        stats = subprocess.Popen(runtime.docker + ["stats", runtime.node, "--format", "{{json .}}"],
                                 stdout=stats_log, stderr=subprocess.STDOUT, env=runtime.env)
        runtime.load_images(hotel.IMAGES)
        os.environ["PATH"] = str(kubectl.parent) + os.pathsep + (old_env["PATH"] or "")
        os.environ["KUBECONFIG"] = str(runtime.kubeconfig)
        install("initial")
        baseline_names = topology("initial")
        usage("initial")
        current_state = snapshot()
        save(output / "initial-state.json", current_state)
        require(hotel.fresh_seed(current_state), "initial reservation seed is not fresh")
        rows = json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-profile", "--", "mongo", "--quiet", "--eval", retry.CATALOG_JS]))
        catalog = {row["id"]: row for row in rows}
        save(output / "catalog.json", rows)
        require(len(rows) == len(catalog) == 80, "profile catalog is incomplete")
        workload = guarded_workload(workload_class, namespace)
        warm("initial")
        booking("preflight")
        entry = phase("healthy")
        workload.start()
        time.sleep(20)
        entry["measurement"] = measure("healthy", 20, retry.healthy, initial=True)
        metrics = entry["measurement"]["after_metrics"]
        require(metrics.get("rate_backend_qps_limit") == 20 and metrics.get("rate_queue_capacity") == 256,
                "runtime policy differs from vulnerable configuration")
        booking("healthy")
        save(output / "summary.json", result)
        entry = phase("sustained")
        entry["burst"] = burst("trigger")
        time.sleep(20)
        entry["measurement"] = measure("sustained", 25, retry.collapsed)
        entry = phase("frontend_restart")
        runtime.kube(["-n", namespace, "rollout", "restart", "deployment/frontend"])
        runtime.kube(["-n", namespace, "rollout", "status", "deployment/frontend", "--timeout=120s"], timeout=135)
        time.sleep(20)
        entry["measurement"] = measure("frontend-restart", 20, retry.collapsed)
        topology("frontend-restart")
        entry = phase("repair")
        for name, env in retry.MITIGATED.items():
            runtime.kube(["-n", namespace, "set", "env", "deployment/" + name, *[key + "=" + value for key, value in env.items()]])
        for name in retry.MITIGATED:
            runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + name, "--timeout=120s"], timeout=135)
        time.sleep(20)
        entry["measurement"] = measure("repair", 20, retry.healthy)
        topology("repair")
        booking("repair")
        entry = phase("replay")
        entry["burst"] = burst("replay")
        time.sleep(20)
        entry["measurement"] = measure("replay", 20, retry.healthy)
        topology("replay")
        booking("replay")
        stop_workload(workload)
        save_driver("protected")
        entry = phase("namespace_reset")
        save(output / "before-reset-state.json", snapshot())
        runtime.kube(["delete", "namespace", namespace, "--wait=true", "--timeout=60s"], timeout=75)
        remaining = json.loads(runtime.kube(["get", "namespace", "-o", "json"]))
        require(namespace not in [obj["metadata"]["name"] for obj in remaining["items"]], "namespace survived reset")
        install("reset")
        topology("reset")
        current_state = snapshot()
        save(output / "reset-state.json", current_state)
        require(hotel.fresh_seed(current_state), "reset did not restore fresh database state")
        workload = guarded_workload(workload_class, namespace)
        warm("reset")
        booking("reset")
        metrics = workload.metrics.snapshot()
        save(output / "reset-metrics.json", metrics)
        require(metrics.get("rate_backend_qps_limit") == 20 and metrics.get("rate_queue_capacity") == 256, "reset leaked repair policy")
        entry["passed"] = True
        save(output / "final-state.json", snapshot())
        usage("final")
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            for resource in ("pods", "events", "deployments"):
                save(output / ("failure-" + resource + ".json"), json.loads(runtime.kube(["-n", namespace, "get", resource, "-o", "json"], timeout=15)))
            usage("failure")
        except Exception as diagnostic_error:
            result["diagnostic_error"] = str(diagnostic_error)
        raise
    finally:
        signal.alarm(0)
        if workload is not None:
            try:
                stop_workload(workload)
                save_driver("final")
            except Exception as exc:
                result["driver_cleanup_error"] = str(exc)
                result["status"] = "failed"
        save(output / "http-events.json", recorder.snapshot())
        try:
            stop_stats(stats)
        except Exception as exc:
            result["stats_cleanup_error"] = str(exc)
            result["status"] = "failed"
        if stats_log is not None:
            stats_log.close()
        cleanup = runtime.close()
        result["cleanup_ok"] = cleanup["ok"]
        if not cleanup["ok"]:
            result["status"] = "failed"
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result["elapsed_s"] = time.monotonic() - started
        save(output / "summary.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "phases"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=Path, default=Path("results/service-kube-runtime-20261021/kind"))
    parser.add_argument("--inputs", type=Path, default=Path("results/service-retry-source-20261021"))
    parser.add_argument("--workload", type=Path, default=Path("results/service-plan-source-20261019/sregym/sregym/generators/workload/hotel_search.py"))
    parser.add_argument("--context", default="desktop-linux")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(1500)
    run(args)


if __name__ == "__main__":
    main()
