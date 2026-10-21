import argparse
import ast
import concurrent.futures
import hashlib
import http.client
import http.server
import json
import signal
import subprocess
import threading
import textwrap
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


REVISION = "482846e48c6cd7ae1a259d8412ae4ebc76a035bd"
SOURCE_SHA256 = "98e81324be7f68d43494dd85655adb23a426fec69fffa333c93de6c663aa9384"
PHASES = (
    ("healthy", 6),
    ("fault", 8),
    ("restart_only", 8),
    ("repaired", 8),
    ("disabled", 6),
    ("false_response", 4),
    ("restored", 6),
)


def serve(mode):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            payload = {"status": "ok"}
            if mode == "backend":
                payload = {
                    "fixture": "application-response",
                    "request_id": query.get("request_id", [None])[0],
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http.server.ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


def filter_program(source):
    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream source hash differs from the frozen protocol")
    tree = ast.parse(body)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_edge_filter_script")
    namespace = {"textwrap": textwrap}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    attributes = types.SimpleNamespace(regex_env="WAF_RULE_REGEX", bad_regex="^([a-zA-Z]+)*$")
    return namespace["_edge_filter_script"](attributes)


def docker(*args, check=True):
    result = subprocess.run(
        ["docker", "--context", "desktop-linux", *args],
        capture_output=True, text=True, timeout=30,
    )
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()}")
    output = result.stdout + result.stderr if args and args[0] == "logs" else result.stdout
    return output.strip()


def probe(url, request_id=None, timeout=0.75):
    started = time.monotonic()
    result = {"http_ok": False, "valid": False, "status": None}
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as response:
            body = response.read(65536)
            result["status"] = response.status
            result["http_ok"] = response.status == 200
        if request_id is not None:
            payload = json.loads(body)
            result["valid"] = (
                isinstance(payload, dict)
                and payload.get("fixture") == "application-response"
                and payload.get("request_id") == request_id
            )
    except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_s"] = time.monotonic() - started
    return result


def cpu_counters(container):
    raw = docker("exec", container, "cat", "/sys/fs/cgroup/cpu.stat")
    return {key: int(value) for key, value in (line.split() for line in raw.splitlines())}


def run_audit(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root / "results" / "service-operator-source" / REVISION / "sregym/conductor/problems/edge_request_filter_cpu_saturation.py"
    script = output / "upstream-filter.pl"
    script.write_text(filter_program(source))
    run_id = "sre-control-" + uuid.uuid4().hex[:10]
    label = "backworld.service-control"
    network = run_id
    backend = run_id + "-backend"
    frontend = run_id + "-frontend"
    images = {
        role: json.loads(docker("image", "inspect", image))[0]
        for role, image in {"perl": "perl:5.36-slim", "python": "python:3.12-slim-bookworm"}.items()
    }
    started = time.monotonic()
    stopped = threading.Event()
    lock = threading.Lock()
    requests = []
    phases = []
    active_phase = {"cycle": 0, "name": "setup", "stage": "transition"}
    config = {
        "scope": "isolated incident mechanism; no kubernetes or agent",
        "protocol_version": 2,
        "network_mode": "dedicated_bridge_with_loopback_ports",
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "program_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": run_id,
        "cycles": 2,
        "phases": PHASES,
        "rate_per_second": 5,
        "request_timeout_s": 0.75,
        "frontend_cpu_limit": 0.1,
        "frontend_memory_mib": 65,
        "images": {role: {"id": value["Id"], "digests": value.get("RepoDigests", [])}
                   for role, value in images.items()},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    live_threads = []
    failure = None
    cleanup_errors = []
    worker_errors = []

    def owned(kind, name):
        raw = docker(kind, "inspect", name, check=False)
        if not raw:
            return False
        values = json.loads(raw)
        if not values:
            return False
        value = values[0]
        labels = value.get("Labels") if kind == "network" else value.get("Config", {}).get("Labels")
        if (labels or {}).get(label) != run_id:
            raise RuntimeError(f"resource ownership mismatch: {name}")
        return True

    def remove_frontend():
        if owned("container", frontend):
            docker("rm", "--force", frontend)

    def launch(name, image, command, memory, cpus, port="", extra=()):
        return docker(
            "run", "--detach", "--name", name, "--label", f"{label}={run_id}",
            "--network", network, "--memory", memory, "--memory-swap", memory,
            "--cpus", str(cpus), "--pids-limit", "32", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--read-only", "--user", "65534:65534",
            "--publish", f"127.0.0.1:{port}:8080",
            "--mount", f"type=bind,source={Path(__file__).resolve()},target=/audit.py,readonly",
            *extra, image, "timeout", "300", *command,
        )

    def collect(url, request_id, phase):
        row = {"request_id": request_id, "started_s": time.monotonic() - started, **phase}
        row.update(probe(url, request_id))
        with lock:
            requests.append(row)

    def normal_traffic(base):
        slots = threading.Semaphore(4)

        def finished(future):
            slots.release()
            error = future.exception()
            if error:
                worker_errors.append(f"{type(error).__name__}: {error}")
                stopped.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            next_arrival = time.monotonic()
            while not stopped.is_set():
                if stopped.wait(max(0, next_arrival - time.monotonic())):
                    break
                request_id = uuid.uuid4().hex
                with lock:
                    phase = dict(active_phase)
                if slots.acquire(blocking=False):
                    future = executor.submit(collect, f"{base}/?request_id={request_id}", request_id, phase)
                    future.add_done_callback(finished)
                else:
                    with lock:
                        requests.append({"request_id": request_id, "started_s": time.monotonic() - started,
                                         **phase, "http_ok": False, "valid": False,
                                         "status": None, "latency_s": None, "error": "client_capacity"})
                next_arrival += 0.2

    def crafted_traffic(base):
        url = base + "/?waf=" + "a" * 5000 + "!"
        while not stopped.is_set():
            probe(url, timeout=10)
            stopped.wait(0.05)

    try:
        docker("network", "create", "--driver", "bridge", "--label", f"{label}={run_id}", network)
        launch(backend, images["python"]["Id"], ["python", "/audit.py", "--serve", "backend"], "96m", 0.25)
        backend_port = docker("port", backend, "8080/tcp").rsplit(":", 1)[-1]
        backend_url = "http://127.0.0.1:" + backend_port
        frontend_port = ""
        for cycle in (1, 2):
            for name, duration in PHASES:
                if worker_errors:
                    raise RuntimeError(f"traffic worker failed: {worker_errors}")
                if time.monotonic() - started > 240:
                    raise TimeoutError("control experiment exceeded its time budget")
                with lock:
                    active_phase.update(cycle=cycle, name=name, stage="transition")
                action_start = time.monotonic()
                remove_frontend()
                if name == "false_response":
                    launch(frontend, images["python"]["Id"], ["python", "/audit.py", "--serve", "false_response"],
                           "96m", 0.25, frontend_port)
                else:
                    rule = "^([a-zA-Z]+)*$" if name in {"fault", "restart_only", "disabled"} else "^[A-Za-z]+$"
                    launch(frontend, images["perl"]["Id"], ["perl", "/filter.pl"], "65m", 0.1, frontend_port, (
                        "--mount", f"type=bind,source={script},target=/filter.pl,readonly",
                        "--env", "FRONTEND_HOST=" + backend,
                        "--env", "FRONTEND_PORT=8080",
                        "--env", "WAF_RULE_REGEX=" + rule,
                        "--env", "WAF_RULE_ENABLED=" + ("false" if name == "disabled" else "true"),
                    ))
                frontend_port = docker("port", frontend, "8080/tcp").rsplit(":", 1)[-1]
                frontend_url = "http://127.0.0.1:" + frontend_port
                if not live_threads:
                    for target in (normal_traffic, crafted_traffic):
                        thread = threading.Thread(target=target, args=(frontend_url,), daemon=True)
                        thread.start()
                        live_threads.append(thread)
                if not phases:
                    config["perl_version"] = docker("exec", frontend, "perl", "-e", "print $^V")
                    config["python_version"] = docker("exec", backend, "python", "--version")
                    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
                before = cpu_counters(frontend)
                window_start = time.monotonic()
                with lock:
                    active_phase.update(stage="stable")
                stopped.wait(duration)
                with lock:
                    active_phase.update(stage="transition")
                window_end = time.monotonic()
                after = cpu_counters(frontend)
                backend_check_id = uuid.uuid4().hex
                direct = probe(backend_url + "/?request_id=" + backend_check_id, backend_check_id)
                row = {
                    "cycle": cycle, "name": name,
                    "action_started_s": action_start - started,
                    "stable_started_s": window_start - started,
                    "stable_ended_s": window_end - started,
                    "action_duration_s": window_start - action_start,
                    "cpu_before": before, "cpu_after": after,
                    "direct_backend": direct,
                    "container_state": json.loads(docker("inspect", "--format", "{{json .State}}", frontend)),
                }
                phases.append(row)
                logs = docker("logs", "--tail", "100", frontend, check=False)
                (output / f"cycle-{cycle}-{name}.log").write_text(logs)
                print(json.dumps({"cycle": cycle, "phase": name, "backend_valid": direct["valid"]}), flush=True)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        stopped.set()
        for thread in live_threads:
            thread.join(timeout=12)
        try:
            for name in (frontend, backend):
                if owned("container", name):
                    (output / f"final-{name.rsplit('-', 1)[-1]}.log").write_text(docker("logs", "--tail", "100", name, check=False))
        except Exception as exc:
            cleanup_errors.append(f"final logs: {exc}")
        for kind, name in (("container", frontend), ("container", backend), ("network", network)):
            try:
                if owned(kind, name):
                    docker("network", "rm", name) if kind == "network" else docker("rm", "--force", name)
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        requests.sort(key=lambda row: row["started_s"])
        (output / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests))
        summaries = []
        for phase in phases:
            rows = [row for row in requests if row["cycle"] == phase["cycle"]
                    and row["name"] == phase["name"] and row["stage"] == "stable"]
            valid_latencies = sorted(row["latency_s"] for row in rows if row["valid"])
            duration = phase["stable_ended_s"] - phase["stable_started_s"]
            used = phase["cpu_after"]["usage_usec"] - phase["cpu_before"]["usage_usec"]
            summaries.append({
                **phase, "requests": len(rows),
                "http_successes": sum(row["http_ok"] for row in rows),
                "valid_responses": sum(row["valid"] for row in rows),
                "cpu_seconds": used / 1_000_000,
                "cpu_fraction_of_one_core": used / 1_000_000 / duration,
                "valid_latency_p50_s": valid_latencies[len(valid_latencies) // 2] if valid_latencies else None,
            })
        summary = {"config": config, "elapsed_s": time.monotonic() - started,
                   "error": failure, "cleanup_errors": cleanup_errors, "phases": summaries,
                   "total_requests": len(requests), "worker_errors": worker_errors}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failure or cleanup_errors:
        raise RuntimeError(json.dumps({"error": failure, "cleanup_errors": cleanup_errors}))
    print(json.dumps({"output": str(output), "total_requests": len(requests), "elapsed_s": summary["elapsed_s"]}), flush=True)


def handle_termination(signum, frame):
    raise SystemExit("terminated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", choices=("backend", "false_response"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve)
    elif args.output:
        signal.signal(signal.SIGTERM, handle_termination)
        run_audit(args.output)
    else:
        parser.error("provide --output or --serve")


if __name__ == "__main__":
    main()
