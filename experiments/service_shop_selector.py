import argparse
import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path

from service_operator import selector, selector_transfer, tools


SOURCES = ["src/service_operator/" + name + ".py" for name in
           ("astronomy_host", "cart", "tools", "workflow", "selector", "selector_transfer", "relay", "shop_selector", "selector_native")]
SOURCES += ["experiments/service_astronomy_host.py", "experiments/service_astronomy_healthy.py", "experiments/service_shop_selector.py"]
TOOLS = {"kind-linux-amd64": "aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d",
         "kubectl-linux-amd64": "8efbb9435132a190920eb65a47a8c1ecf755ad85ab57a600c9bedbab460bb7a8"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(inputs, injector, output):
    source = json.loads((inputs / "bundle.json").read_text())
    for name, expected in source["files_sha256"].items():
        if sha(inputs / name) != expected:
            raise ValueError("input bundle file changed: " + name)
    for name, expected in TOOLS.items():
        if sha(inputs / name) != expected:
            raise ValueError("Linux binary differs from the pin")
    selector.load_wrong_selector_functions(injector)
    run_id = uuid.uuid4().hex[:16]
    namespace, private = "bw-astro-" + run_id[:10], "bw-check-" + run_id[:10]
    replacements = {source["run_id"]: run_id, source["application_namespace"]: namespace, source["private_namespace"]: private}

    def rewrite(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k,v in value.items()}
        return value

    app = rewrite(json.loads((inputs / "application.json").read_text()))
    base_control = rewrite(json.loads((inputs / "control.json").read_text()))["items"]
    frontend = [o for o in app["items"] if o["kind"] == "Deployment" and o["metadata"]["name"] == "frontend"]
    services = [o for o in app["items"] if o["kind"] == "Service" and o["metadata"]["name"] == "frontend"]
    if len(frontend) != 1 or len(services) != 1:
        raise ValueError("frontend Deployment/Service is missing or duplicated")
    labels, selected = frontend[0]["spec"]["template"]["metadata"]["labels"], services[0]["spec"]["selector"]
    selector_transfer.deployment_selector(frontend[0]["spec"]["selector"])
    tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": labels}}, "selector_shop")
    if not selected or any(labels.get(k) != v for k,v in selected.items()) or labels.get("current_service_name") == "frontend":
        raise ValueError("frontend selector is already invalid or pinned fault would be ineffective")
    editors = [c for o in app["items"] if o["kind"] == "Deployment" and o["metadata"]["name"] == "flagd"
               for c in o["spec"]["template"]["spec"]["containers"] if c["name"] == "flagd-ui"]
    if len(editors) != 1 or editors[0].get("command") != ["sh", "-c", "ulimit -n 65536 && exec /app/bin/server"] or editors[0]["resources"]["limits"]["memory"] != "250Mi":
        raise ValueError("use the checked native editor profile")
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(__file__).resolve().parents[1]
    for name in SOURCES:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / name, destination)
    for name in TOOLS:
        shutil.copyfile(inputs / name, output / name)
        (output / name).chmod(0o755)
    shutil.copyfile(inputs / "image-lock.json", output / "image-lock.json")
    shutil.copyfile(injector, output / "private-injector.py")
    config = next(o for o in base_control if o["kind"] == "ConfigMap")
    names = ("cart", "tools", "workflow", "selector_transfer", "shop_selector", "relay")
    config["data"] = {name + ".py": (workspace / "src/service_operator" / (name + ".py")).read_text() for name in names}
    job = next(o for o in base_control if o["kind"] == "Job")
    job["metadata"]["name"] = "selector-control"
    job["spec"]["activeDeadlineSeconds"] = 600
    pod = job["spec"]["template"]["spec"]
    pod.pop("initContainers", None)
    pod["activeDeadlineSeconds"] = 600
    container = pod["containers"][0]
    container.update(name="source", command=["python", "-B", "-m", "service_operator.shop_selector"],
                     args=["--namespace", namespace, "--run-id", run_id])
    container["readinessProbe"] = {"tcpSocket": {"port": 8081}, "periodSeconds": 1}
    relay = copy.deepcopy(container)
    relay.update(name="relay", command=["python", "-B", "/code/service_operator/relay.py"],
                 args=[f"http://frontend-proxy.{namespace}.svc.cluster.local:8080", "shop"])
    relay["readinessProbe"]["tcpSocket"]["port"] = 8080
    pod["containers"] = [container, relay]
    volume = next(v for v in pod["volumes"] if v["name"] == "code")
    volume["configMap"]["items"] = [{"key": name + ".py", "path": "service_operator/" + name + ".py"} for name in names]
    role = tools.operator_role(namespace, "selector_shop")
    for obj in role:
        obj["metadata"].setdefault("labels", {})["backworld.run"] = run_id
    control = [next(o for o in base_control if o["kind"] == "Namespace"), config, job, *role]
    save(output / "application.json", app)
    save(output / "control.json", {"apiVersion": "v1", "kind": "List", "items": control})
    save(output / "public-task.json", selector_transfer.public_task("shop"))
    plan = {"conditions": list(selector_transfer.CONDITIONS), "expected_grades": [True, True, False, False],
            "fresh_namespace_and_storage_per_condition": True, "source_interval_s": 0.5,
            "action_s": 300, "fixed_post_action_horizon_s": 30, "sustained_recovery_window_s": 10,
            "traffic_protocol_version": 2, "drain_s": 5,
            "fault_effect": "two admitted invalid catalog requests before actor opening; stop if ineffective; no proxy reset",
            "accounting": "all 0.5s source slots until actor opening plus330s, followed by a declared5s drain; each offer retains its full5s deadline; late/missing/undelivered/invalid fail",
            "calibration": "2RPS offered; actual relay admissions and dispatch lag must be measured by the native control",
            "public_boundary": "JSON tools only; no host/filesystem/network/RPC/injector/private namespace access",
            "baseline": "frozen public-only selector investigate() runbook; Hotel native controls remain archived regressions",
            "cloud_execution_authorized": False, "live_controls_passed": False}
    save(output / "private-plan.json", plan)
    files = SOURCES + list(TOOLS) + ["application.json", "control.json", "image-lock.json", "private-injector.py", "public-task.json", "private-plan.json"]
    manifest = {**source, "status": "runnable local package; native selector controls not executed", "run_id": run_id,
                "application_namespace": namespace, "private_namespace": private, "control_mode": "shop_selector_v1", "cycles": 4,
                "input_bundle_sha256": sha(inputs / "bundle.json"), "input_application_sha256": sha(inputs / "application.json"),
                "traffic_protocol_version": 2, "drain_s": 5,
                "files_sha256": {name: sha(output / name) for name in files},
                "source_hashes": {name + ".py": sha(workspace / "src/service_operator" / (name + ".py"))
                                  for name in ("cart", "selector_transfer", "shop_selector")},
                "limits": {"work_s": 5700, "cleanup_s": 300, "startup_s_per_cycle": 900,
                           "node_memory_bytes": 24 * 1024 ** 3, "node_cpus": 6, "node_pids": 8192,
                           "helper_deadline_s": 600, "action_s": 300, "source_interval_s": 0.5},
                "runnable": True, "live_controls_passed": False, "cloud_execution_authorized": False}
    save(output / "bundle.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--injector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs.resolve(), args.injector.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
