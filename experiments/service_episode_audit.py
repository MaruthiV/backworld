import argparse
import hashlib
import json
import signal
import time
import types
import uuid
from pathlib import Path

from service_operator import episode, hotel, http_probe, kubernetes, selector, tools, traffic, workflow
from service_operator.kubernetes import LocalCluster, save
from service_selector_audit import InjectorAdapter


TASK = "Investigate this booking alert. Keep existing reservations intact and decide whether changes are needed."
SCRIPTS = {
    "healthy": [
        ("inspect", {"resource": "service", "name": "frontend"}),
        ("inspect", {"resource": "endpoints", "name": "frontend"}),
        ("inspect", {"resource": "deployment", "name": "frontend"}),
        ("logs", {"deployment": "frontend", "lines": 10}),
        ("check", {}),
        ("finish", {"summary": "booking looks healthy; no changes needed"}),
    ],
    "repair": [
        ("inspect", {"resource": "service", "name": "frontend"}),
        ("inspect", {"resource": "endpoints", "name": "frontend"}),
        ("inspect", {"resource": "pods", "name": "search"}),
        ("logs", {"deployment": "frontend", "lines": 10}),
        ("restart", {"deployment": "frontend"}),
        ("wait", {"seconds": 2}),
        ("set_selector", {"service": "frontend", "selector": {"io.kompose.service": "frontend"}}),
        ("check", {}),
        ("finish", {"summary": "corrected the frontend selector; check recovery"}),
    ],
    "premature_finish": [("finish", {"summary": "investigation complete"})],
}


def record_jsonl(path, event):
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def replay(send, script):
    for name, arguments in script:
        call = json.loads(json.dumps({"tool": name, "arguments": arguments}))
        response = send(call)
        if not response["ok"]:
            raise RuntimeError("scripted public tool call failed")


def node_usage(runtime):
    return {name: runtime.command(runtime.docker + ["exec", runtime.node, "cat", "/sys/fs/cgroup/" + name])
            for name in ("memory.events", "memory.peak", "cpu.stat", "pids.peak")}


def audit_permissions(runtime, backend, private_namespace, path):
    cases = [
        ("get", "services", backend.namespace, True),
        ("patch", "services", backend.namespace, True),
        ("patch", "deployments.apps", backend.namespace, True),
        ("list", "pods", private_namespace, False),
        ("get", "secrets", backend.namespace, False),
        ("create", "pods/exec", backend.namespace, False),
        ("get", "nodes", None, False),
        ("delete", "deployments.apps", backend.namespace, False),
        ("patch", "rolebindings.rbac.authorization.k8s.io", backend.namespace, False),
    ]
    checks = []
    for verb, resource, namespace, expected in cases:
        args = [*backend.identity, "auth", "can-i", verb, resource]
        if namespace:
            args += ["-n", namespace]
        answer = runtime.kube(args, check=False)
        checks.append({"verb": verb, "resource": resource, "namespace": namespace,
                       "answer": answer, "expected": expected, "passed": answer == ("yes" if expected else "no")})
        save(path, checks)
    runtime.kube([*backend.identity, "-n", private_namespace, "get", "pods", "-o", "json"], check=False)
    command = runtime.output / f"command-{runtime.count:04d}"
    metadata = json.loads(command.with_suffix(".json").read_text())
    message = command.with_suffix(".log").read_text()
    checks.append({"operation": "actual private pod read", "returncode": metadata["returncode"],
                   "passed": metadata["returncode"] == 1 and "Error from server (Forbidden): pods is forbidden:" in message
                   and private_namespace in message})
    save(path, checks)
    if not all(check["passed"] for check in checks):
        raise RuntimeError("native operator permissions do not match the boundary")


def run(output, kind, injector_path, context, conditions=None):
    output.mkdir(parents=True, exist_ok=False)
    runtime = LocalCluster(output, kind, context)
    started = time.monotonic()
    result = {"status": "running", "run_id": runtime.name, "episodes": [], "cleanup_ok": False}
    functions = selector.load_selector_functions(injector_path)
    sources = [Path(__file__), Path(__file__).with_name("service_selector_audit.py"), injector_path,
               *[Path(module.__file__) for module in (episode, hotel, http_probe, kubernetes, selector, tools, traffic, workflow)]]
    source_dir = output / "source"
    source_dir.mkdir()
    hashes = {}
    for source in sources:
        body = source.read_bytes()
        (source_dir / source.name).write_bytes(body)
        hashes[source.name] = hashlib.sha256(body).hexdigest()
    images = {"client": hotel.CLIENT_IMAGE, **hotel.IMAGES}
    scripts = {name: SCRIPTS[name] for name in conditions or SCRIPTS}
    save(output / "protocol.json", {
        "version": "2.1", "scope": "scripted JSON tool and episode controls; no model, learning or official score",
        "source_sha256": hashes, "images": images, "node_image": kubernetes.NODE_IMAGE,
        "kind_sha256": kubernetes.KIND_SHA256, "scripts": scripts,
        "scheduled_requests_per_episode": 12, "interval_s": 3, "horizon_s": 36,
        "http_deadline_s": 1, "offered_to_ack_deadline_s": 2, "settle_s": 0.25,
        "clock": "private pod scheduler and completion receipt; database observation outside request loop",
        "state_observations": "initial, post-baseline, immediate/settled after fixed horizon; batch exact-write accounting",
        "overall_deadline_s": 1500, "max_tool_calls": 16, "max_diagnostic_checks": 2,
        "baseline_requests_per_episode": 1,
        "planned_diagnostic_requests": sum(name in scripts for name in ("healthy", "repair")),
        "collection": "wait for fixed horizon; bounded retries of read-only status only; no mutation/write retries",
        "node_limits": {"memory_gib": 4, "cpus": 4, "pids": 2048},
        "upstream_revision": "482846e48c6cd7ae1a259d8412ae4ebc76a035bd",
        "application_revision": "2b2f9c6c2e97c44abbfcc44af1cf2f994bbb04f8",
        "boundary": "JSON-only trusted broker; RBAC namespaces/resources plus gateway field restrictions; not an arbitrary-code sandbox",
        "pass": "healthy and public repair succeed; premature finish fails; complete workloads, preserved state and clean reset/teardown",
    })

    def wait_route(namespace, expected, path):
        deadline = time.monotonic() + 30
        observations = []
        while time.monotonic() < deadline:
            state = episode.routing_state(runtime, namespace)
            observations.append(state)
            save(path, observations)
            if selector.check_route(state, expected):
                return state
            time.sleep(1)
        raise RuntimeError("endpoint identity did not converge")

    try:
        print("starting isolated Kubernetes node", flush=True)
        runtime.start()
        runtime.load_images(images)
        for index, (condition, script) in enumerate(scripts.items(), 1):
            print("starting episode " + str(index), flush=True)
            folder = output / f"episode-{index}"
            folder.mkdir()
            namespace = "app-" + uuid.uuid4().hex[:10]
            private_namespace = "observer-" + uuid.uuid4().hex[:10]
            entry = {"condition": condition, "namespace": namespace, "private_namespace": private_namespace}
            result["episodes"].append(entry)
            runtime.apply([{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}}
                           for name in (namespace, private_namespace)])
            objects = [obj for obj in hotel.kubernetes_profile(namespace, runtime.aliases)
                       if obj["metadata"]["name"] not in ("client", "false-response")]
            client = hotel.deployment("client", "client", runtime.aliases["client"], "128Mi",
                                      ["python", "/opt/operator/traffic.py", f"http://frontend.{namespace}.svc.cluster.local:5000"])
            client["metadata"]["namespace"] = private_namespace
            client["spec"]["template"]["spec"]["volumes"] = [{"name": "traffic-code", "configMap": {"name": "traffic-code"}}]
            client["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [
                {"name": "traffic-code", "mountPath": "/opt/operator", "readOnly": True}]
            code = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "traffic-code", "namespace": private_namespace},
                    "data": {"traffic.py": Path(traffic.__file__).read_text(), "http_probe.py": Path(http_probe.__file__).read_text()}}
            save(folder / "application.json", objects)
            save(folder / "private-client.json", client)
            apps = {"frontend", "search", "reservation", "user"}
            support = [obj for obj in objects if obj["kind"] != "Deployment" or obj["metadata"]["name"] not in apps]
            runtime.apply([*support, code, client, *tools.operator_role(namespace)])
            for obj in [*support, client]:
                if obj["kind"] == "Deployment":
                    runtime.kube(["-n", obj["metadata"]["namespace"], "rollout", "status",
                                  "deployment/" + obj["metadata"]["name"], "--timeout=90s"], timeout=100)
            runtime.apply([obj for obj in objects if obj["kind"] == "Deployment" and obj["metadata"]["name"] in apps])
            for name in sorted(apps):
                runtime.kube(["-n", namespace, "rollout", "status", "deployment/" + name, "--timeout=90s"], timeout=100)
            ledger = episode.BookingLedger(runtime, namespace, private_namespace, folder / "private-ledger")
            entry["fresh_seed"] = hotel.fresh_seed(ledger.before)
            if not entry["fresh_seed"]:
                raise RuntimeError("application reset has invalid seed state")
            ready = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                response = ledger.http(f"http://frontend.{namespace}.svc.cluster.local:5000/user?username=Cornell_1&password=1111111111")
                ready.append(response)
                save(folder / "initial-readiness.json", ready)
                if response.get("body") == {"message": "Login successfully!"}:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("frontend login did not become ready")
            if not ledger.probe("baseline")["verdict"]["accepted"]:
                raise RuntimeError("baseline booking was not confirmed")
            if condition != "healthy":
                adapter = InjectorAdapter(runtime, namespace, folder / "injector-actions.jsonl")
                target = types.SimpleNamespace(kubectl=adapter, namespace=namespace)
                functions["inject_service_wrong_pod_selection"](target, ["frontend", "search"])
            wait_route(namespace, ["frontend"] if condition == "healthy" else ["frontend", "search"],
                       folder / "initial-routing.json")
            backend = tools.OperatorBackend(runtime, namespace)
            audit_permissions(runtime, backend, private_namespace, folder / "permissions.json")
            save(folder / "public-task.json", {"task": TASK, "tools": tools.tool_definitions(),
                                              "tool_budget": 16, "diagnostic_check_budget": 2})
            save(folder / "node-before.json", node_usage(runtime))
            worker = episode.PrivateWorkload(ledger)
            worker.start()
            public = tools.PublicTools(backend, ledger.public_check,
                                       lambda event: record_jsonl(folder / "public-tools.jsonl", event), worker.started,
                                       complete=ledger.complete)
            actor_error = None
            try:
                replay(public.invoke, script)
            except Exception as exc:
                actor_error = f"{type(exc).__name__}: {exc}"
            finally:
                worker.join()
            save(folder / "scheduled-workload.json", {"started_monotonic_s": worker.driver_state["started_monotonic_s"],
                 "ended_monotonic_s": worker.driver_state["ended_monotonic_s"],
                 "declared_complete_monotonic_s": worker.driver_state["declared_complete_monotonic_s"],
                 "events": worker.events, "error": worker.error})
            save(folder / "node-after.json", node_usage(runtime))
            final = episode.routing_state(runtime, namespace)
            save(folder / "final-routing.json", final)
            route_ok = selector.check_route(final, ["frontend"])
            grade = episode.grade_episode(worker, ledger, public, route_ok, condition == "healthy")
            grade.update({"tool_calls": public.calls, "diagnostic_checks": public.checks,
                          "finished_at_monotonic_s": public.finished_at, "actor_error": actor_error})
            save(folder / "private-grade.json", grade)
            expected_success = condition != "premature_finish"
            passed = (actor_error is None and grade["workload_complete"] and grade["invariants_preserved"]
                      and grade["post_completion_requests"] >= 2 and grade["success"] == expected_success
                      and (expected_success or not route_ok))
            entry.update({"grade": grade, "passed": passed})
            save(output / "summary.json", result)
            print(f"episode {index}: {grade['scheduled_valid']}/12 valid; success={grade['success']}; control={passed}", flush=True)
            if not passed:
                raise RuntimeError("episode failed its frozen control rule")
            runtime.kube(["delete", "namespace", namespace, private_namespace, "--wait=true", "--timeout=75s"], timeout=85)
            remaining = json.loads(runtime.kube(["get", "namespaces", "-o", "json"]))
            names = {item["metadata"]["name"] for item in remaining["items"]}
            entry["reset_ok"] = not names.intersection((namespace, private_namespace))
            if not entry["reset_ok"]:
                raise RuntimeError("owned episode namespaces remained after reset")
            save(output / "summary.json", result)
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
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=Path, default=Path("results/service-kube-runtime-20261021/kind"))
    parser.add_argument("--injector", type=Path, default=Path("results/service-kube-runtime-20261021/inject_virtual.py"))
    parser.add_argument("--context", default="desktop-linux")
    parser.add_argument("--conditions", nargs="+", choices=list(SCRIPTS))
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(1500)
    run(args.output.resolve(), args.kind, args.injector, args.context, args.conditions)


if __name__ == "__main__":
    main()
