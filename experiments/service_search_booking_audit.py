import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from service_operator import hotel, http_probe, kubernetes, retry, workflow
from service_operator.kubernetes import LocalCluster, save
from service_retry_audit import INPUTS, KUBECTL_SHA256, guarded_workload, stop_workload


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime = LocalCluster(output, Path("results/service-kube-runtime-20261021/kind"))
    inputs = Path("results/service-retry-source-20261021")
    source_dir = output / "source"
    source_dir.mkdir()
    for path in [Path(__file__), Path(__file__).with_name("service_retry_audit.py"),
                 *[Path(module.__file__) for module in (hotel, http_probe, kubernetes, retry, workflow)]]:
        (source_dir / path.name).write_bytes(path.read_bytes())
    for relative, expected in INPUTS.items():
        body = (inputs / relative).read_bytes()
        if hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() != expected:
            raise ValueError("pinned input changed")
    kubectl = inputs.resolve() / "bin/kubectl"
    if hashlib.sha256(kubectl.read_bytes()).hexdigest() != KUBECTL_SHA256:
        raise ValueError("kubectl pin changed")
    protocol = Path("docs/SERVICE_SEARCH_BOOKING_PROTOCOL.md").read_bytes()
    (output / "protocol.txt").write_bytes(protocol)
    recorder = retry.RecordedRequests(output / "metrics.jsonl")
    workload_path = Path("results/service-plan-source-20261019/sregym/sregym/generators/workload/hotel_search.py")
    workload_class = retry.load_workload(workload_path, recorder)
    (source_dir / workload_path.name).write_bytes(workload_path.read_bytes())
    save(output / "source-sha256.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_dir.iterdir()})
    namespace = "booking-cache"
    result = {"status": "running", "run_id": runtime.name, "bookings": [], "searches": [], "cleanup_ok": False}
    workload = None
    cache_forward = None
    old_env = {key: os.environ.get(key) for key in ("PATH", "KUBECONFIG")}

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def snapshot():
        return json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-reservation", "--",
                                       "mongo", "--quiet", "--eval", hotel.SNAPSHOT_JS]))

    def search(label):
        port = workload.frontend.start()
        response = http_probe.request(f"http://127.0.0.1:{port}" + workload._PATH, 4)
        result["searches"].append({"label": label, "unix_s": time.time(), "response": response})
        save(output / "summary.json", result)
        return response

    def ids(response):
        if isinstance(response.get("body"), dict) and response["body"].get("features") == []:
            require(response["status"] == 200 and response["elapsed_s"] <= 4, "empty result had transport failure")
            return []
        retry.select_hotel(response, catalog)
        return [feature["id"] for feature in response["body"]["features"]]

    def cache(label, clear=False):
        port = cache_forward.start()
        keys = [key + suffix for key in original for suffix in ("_cap", "_2015-04-10_2015-04-10")]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
            if clear:
                replies = []
                for key in keys:
                    connection.sendall(("delete " + key + "\r\n").encode())
                    reply = b""
                    while not reply.endswith(b"\r\n"):
                        part = connection.recv(128)
                        require(bool(part) and len(reply) < 128, "cache delete reply failed")
                        reply += part
                    replies.append({"key": key, "reply": reply.decode()})
                save(output / "cache-deletions.json", replies)
                require(all(row["reply"] == "DELETED\r\n" for row in replies), "expected cache keys were not deleted")
            connection.sendall(("get " + " ".join(keys) + "\r\n").encode())
            body = b""
            while not body.endswith(b"END\r\n"):
                part = connection.recv(4096)
                require(bool(part) and len(body) < 16384, "cache read failed")
                body += part
        (output / (label + "-cache.txt")).write_bytes(body)

    def book(selected, label):
        nonlocal state
        expected = {"hotelId": selected, "customerName": runtime.name + "-" + label,
                    "inDate": "2015-04-09", "outDate": "2015-04-10", "number": 1}
        event = {"label": label, "expected": expected, "before": state, "unix_s": time.time()}
        path = output / (label + "-booking.json")
        save(path, event)
        port = workload.frontend.start()
        event["response"] = http_probe.request(f"http://127.0.0.1:{port}/reservation?" + urlencode(
            {**expected, "username": "Cornell_1", "password": "1111111111"}), 4)
        save(path, event)
        event["immediate"] = snapshot()
        time.sleep(0.25)
        event["after"] = snapshot()
        event["verdict"] = workflow.check_booking(state["reservations"], event["after"]["reservations"], expected, event["response"])
        save(path, event)
        require(event["verdict"]["accepted"] and event["immediate"] == event["after"]
                and state["capacity"] == event["after"]["capacity"], "invalid booking; not retried")
        state = event["after"]
        result["bookings"].append(event["verdict"])

    try:
        runtime.start()
        runtime.load_images(hotel.IMAGES)
        os.environ["PATH"] = str(kubectl.parent) + os.pathsep + (old_env["PATH"] or "")
        os.environ["KUBECONFIG"] = str(runtime.kubeconfig)
        runtime.kube(["create", "namespace", namespace])
        objects = retry.profile(namespace, runtime.aliases, {name: (inputs / f"applications/hotelReservation/k8s-{name}-mongo.sh").read_text() for name in ("geo", "rate")})
        save(output / "manifest.json", objects)
        runtime.apply(objects)
        deadline = time.monotonic() + 180
        for obj in objects:
            if obj["kind"] == "Deployment":
                remaining = max(1, int(deadline - time.monotonic()))
                require(time.monotonic() < deadline, "rollout deadline")
                runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + obj["metadata"]["name"], f"--timeout={remaining}s"], timeout=remaining + 10)
        state = snapshot()
        require(hotel.fresh_seed(state), "database not fresh")
        save(output / "initial-state.json", state)
        save(output / "initial-pods.json", json.loads(runtime.kube(["-n", namespace, "get", "pods", "-o", "json"])))
        rows = json.loads(runtime.kube(["-n", namespace, "exec", "deployment/mongodb-profile", "--", "mongo", "--quiet", "--eval", retry.CATALOG_JS]))
        catalog = {row["id"]: row for row in rows}
        save(output / "catalog.json", rows)
        require(len(catalog) == 80, "incomplete catalog")
        workload = guarded_workload(workload_class, namespace)
        deadline = time.monotonic() + 20
        while True:
            response = search("warmup")
            try:
                original = ids(response)
                require(len(original) == 2, "initial candidate set differs from the diagnosed two-hotel case")
                break
            except ValueError:
                require(time.monotonic() < deadline, "search warmup deadline")
                time.sleep(0.5)
        cache_forward = sys.modules[workload_class.__module__].KubectlPortForward(namespace, "memcached-reserve", 11211)
        cache("before")
        for i in range(3):
            require(set(ids(search("healthy-" + str(i)))) == set(original), "healthy search was unstable")
        book(original[0], "first")
        cache("after-first")
        remaining_ids = []
        for i in range(3):
            remaining_ids = ids(search("after-first-" + str(i)))
            require(set(remaining_ids) == set(original[1:]), "first booked hotel was not uniquely filtered")
        book(remaining_ids[0], "second")
        cache("after-second")
        for i in range(3):
            require(ids(search("after-second-" + str(i))) == [], "two one-room bookings did not cause empty search")
        require(snapshot() == state, "search modified stored state")
        cache("after-cache-delete", clear=True)
        restored = []
        for i in range(3):
            restored.append(set(ids(search("cache-cleared-" + str(i)))) == set(original))
        cache("after-cleared-searches")
        require(snapshot() == state, "cache deletion changed stored bookings or capacity")
        require(all(restored), "cache deletion did not restore both hotels")
        save(output / "final-state.json", snapshot())
        save(output / "final-metrics.json", workload.metrics.snapshot())
        save(output / "final-pods.json", json.loads(runtime.kube(["-n", namespace, "get", "pods", "-o", "json"])))
        usage = {name: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + name]) for name in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
        save(output / "resources.json", usage)
        require(dict(line.split() for line in usage["memory.events"].splitlines())["oom_kill"] == "0", "node OOM kill")
        result["original_ids"] = original
        result["status"] = "hypothesis_confirmed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        try:
            if workload is not None:
                save(output / "collected-final-state.json", snapshot())
                save(output / "collected-final-pods.json", json.loads(runtime.kube(["-n", namespace, "get", "pods", "-o", "json"], timeout=15)))
                usage = {name: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + name], timeout=10)
                         for name in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}
                save(output / "collected-resources.json", usage)
        except Exception as exc:
            result["collection_error"] = str(exc)
        if workload is not None:
            stop_workload(workload)
        if cache_forward is not None:
            cache_forward.stop()
        result["cleanup_ok"] = runtime.close()["ok"]
        if not result["cleanup_ok"]:
            result["status"] = "failed"
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result["elapsed_s"] = time.monotonic() - started
        save(output / "summary.json", result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("bookings", "searches")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(600)
    run(args.output.resolve())


if __name__ == "__main__":
    main()
