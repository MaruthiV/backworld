import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from service_operator import selector, selector_transfer, tools


def prepare(application, objects_path, injector_path, namespace, output):
    if application not in ("hotel", "shop") or not re.fullmatch(r"bw-(astro|hotel)-[a-f0-9]{10}", namespace):
        raise ValueError("use a named application and fresh owned namespace")
    selector.load_wrong_selector_functions(injector_path)
    document = json.loads(objects_path.read_text())
    objects = document["items"] if isinstance(document, dict) else document
    deployments = [row for row in objects if row["kind"] == "Deployment" and row["metadata"]["name"] == "frontend"]
    services = [row for row in objects if row["kind"] == "Service" and row["metadata"]["name"] == "frontend"]
    if len(deployments) != 1 or len(services) != 1:
        raise ValueError("input must contain one frontend Deployment and Service")
    deployment, service = deployments[0], services[0]
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    initial = service["spec"].get("selector")
    if (service["spec"].get("type", "ClusterIP") != "ClusterIP" or not initial
            or any(labels.get(key) != value for key, value in initial.items())
            or labels.get("current_service_name") == "frontend"):
        raise ValueError("frontend baseline or pinned fault precondition is invalid")
    selector_transfer.deployment_selector(deployment["spec"]["selector"])
    profile = "selector_" + application
    tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": initial}}, profile)
    tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": labels}}, profile)
    context = selector_transfer.public_task(application)
    role = tools.operator_role(namespace, profile)
    private = {"application": application, "namespace": namespace, "target": "frontend",
               "conditions": [{"id": condition, "inject_fault": condition != "healthy",
                               "actor": "public_runbook" if condition in ("healthy", "known_repair") else condition,
                               "expected_grade": condition in ("healthy", "known_repair"),
                               "fresh_namespace_and_state": True} for condition in selector_transfer.CONDITIONS],
               "injector_git_blob": selector.INJECTOR_BLOB,
               "inject_method": "inject_wrong_service_selector", "recovery_method": "recover_wrong_service_selector",
               "original_selector": initial, "injected_selector": {**initial, "current_service_name": "frontend"},
               "fault_effect_gate": "require actual failed public workflow before actor time; empty endpoints alone do not establish an outage",
               "state_gate": "independent exact stored-write/expiry/prior-state checks; no uncertain-write retry",
               "clock": "all offers, outcomes, finish and horizon in one source monotonic clock domain",
               "loss": "all source offers in the full fault-to-horizon interval; missing, undelivered, invalid and late outcomes fail",
               "finish": "protected work continues; require at least two post-completion requests and sustained recovery",
               "public_check": "application plus ordered responses with status/body/elapsed_s/deadline_s; shop also returns its normal POST intent",
               "public_data_exclusions": ["case identity", "injection machinery", "expected selector", "private store snapshots", "grade"],
               "native_ready": False,
               "remaining_gates": ["integrate native workload/relay and private store adapters", "validate actual injected effect and all four controls",
                                   "measure/freeze delivered workload and whole-attempt horizon", "separate scoped CPU authorization"]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("public-task.json", context), ("operator-rbac.json", role), ("private-control-plan.json", private)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(injector_path, output / "private-injector.py")
    sources = [Path(tools.__file__), Path(selector.__file__), Path(selector_transfer.__file__), Path(__file__)]
    manifest = {"status": "local selector contract/control preparation; not a live run", "application": application,
                "namespace": namespace, "input_objects_sha256": hashlib.sha256(objects_path.read_bytes()).hexdigest(),
                "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources},
                "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()},
                "paid_execution": False, "native_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--application", choices=("hotel", "shop"), required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--injector", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.application, args.objects, args.injector, args.namespace, args.output), indent=2))


if __name__ == "__main__":
    main()
