import base64
import copy
import importlib.util
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

import pytest

from service_operator import astronomy_host as host


def module(name):
    path = Path(__file__).parents[1] / "experiments" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


probe, runner = module("service_astronomy_healthy.py"), module("service_astronomy_host.py")
RUN = "0123456789abcdef"
NAMESPACE = "bw-astro-0123456789"
PREFIX = ["kubectl", "--kubeconfig", "private-file", "--context", "kind-owned"]
PRODUCT = {"id": "sku", "name": "Telescope", "picture": "/image.png", "description": "A telescope",
           "categories": ["telescopes"], "priceUsd": {"currencyCode": "USD", "units": 19, "nanos": 0}}


def planned(kind="Namespace", name=NAMESPACE):
    version = "v1" if kind == "Namespace" else "rbac.authorization.k8s.io/v1"
    return {"apiVersion": version, "kind": kind, "metadata": {"name": name, "labels": {"backworld.run": RUN}}}


class FakeAPI:
    def __init__(self):
        self.objects, self.calls, self.counter = {}, [], 0
        self.timeout_create = self.replace_before_delete = False

    def __call__(self, args, input=None):
        self.calls.append((args, input))
        operation = args[len(PREFIX)]
        if operation == "get":
            resource, name = args[len(PREFIX) + 1:len(PREFIX) + 3]
            actual = next((v for v in self.objects.values() if host.identity(v)["resource"] == resource and v["metadata"]["name"] == name), None)
            return json.dumps(actual) if actual else ""
        if operation == "create":
            value = json.loads(input)
            self.counter += 1
            value["metadata"]["uid"] = "created-" + str(self.counter)
            self.objects[host.identity(value)["key"]] = value
            if self.timeout_create:
                raise TimeoutError("create committed but response was lost")
            return json.dumps(value)
        if operation == "delete":
            path = args[args.index("--raw") + 1]
            key, value = next((k,v) for k,v in self.objects.items() if host.identity(v)["path"] == path)
            if self.replace_before_delete:
                value["metadata"]["uid"] = "replacement"
            if json.loads(input)["preconditions"]["uid"] != value["metadata"]["uid"]:
                raise RuntimeError("409 UID precondition failed")
            del self.objects[key]
            return "{}"
        raise AssertionError(args)


def test_partial_create_is_not_retried_and_remains_cleanup_eligible(tmp_path):
    api = FakeAPI()
    api.timeout_create = True
    obj = planned()
    actor = host.KubeControl(api, PREFIX, [obj], tmp_path, RUN)
    with pytest.raises(TimeoutError):
        actor.create([obj])
    assert sum(args[len(PREFIX)] == "create" for args,_ in api.calls) == 1
    assert actor.receipts[host.identity(obj)["key"]]["uncertain_create"]
    assert actor.cleanup([obj])["ok"]
    assert not api.objects


def test_preexisting_objects_are_never_adopted(tmp_path):
    api = FakeAPI()
    obj = planned()
    api.objects[host.identity(obj)["key"]] = {**obj, "metadata": {**obj["metadata"], "uid": "preexisting"}}
    actor = host.KubeControl(api, PREFIX, [obj], tmp_path, RUN)
    with pytest.raises(ValueError, match="already exists"):
        actor.create([obj])
    assert not actor.intents and not actor.receipts
    assert all(args[len(PREFIX)] != "create" for args,_ in api.calls)


@pytest.mark.parametrize("damage", ["owner", "uid", "race"])
def test_cleanup_refuses_replaced_resources_and_uses_api_uid_preconditions(tmp_path, monkeypatch, damage):
    api, obj = FakeAPI(), planned()
    actor = host.KubeControl(api, PREFIX, [obj], tmp_path, RUN)
    actor.create([obj])
    actual = api.objects[host.identity(obj)["key"]]
    if damage == "owner":
        actual["metadata"]["labels"]["backworld.run"] = "someone-else"
    elif damage == "uid":
        actual["metadata"]["uid"] = "replacement"
    else:
        api.replace_before_delete = True
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr(host.time, "monotonic", lambda: next(ticks))
    result = actor.cleanup([obj])
    assert not result["ok"] and api.objects
    deletes = [json.loads(body) for args,body in api.calls if args[len(PREFIX)] == "delete"]
    assert deletes == ([{"apiVersion": "v1", "kind": "DeleteOptions", "preconditions": {"uid": "created-1"}, "propagationPolicy": "Foreground"}] if damage == "race" else [])


def test_cleanup_includes_cluster_rbac_not_only_namespaces(tmp_path):
    api = FakeAPI()
    objects = [planned(), planned("Namespace", "bw-check-0123456789"),
               planned("ClusterRole", "otel-collector"), planned("ClusterRoleBinding", "otel-collector")]
    actor = host.KubeControl(api, PREFIX, objects, tmp_path, RUN)
    actor.create(objects)
    assert actor.cleanup(objects)["ok"]
    assert not api.objects
    paths = [args[args.index("--raw") + 1] for args,_ in api.calls if args[len(PREFIX)] == "delete"]
    assert "/apis/rbac.authorization.k8s.io/v1/clusterroles/otel-collector" in paths
    assert "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings/otel-collector" in paths


def test_no_default_context_or_outside_package_reads(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        host.KubeControl(FakeAPI(), ["kubectl"], [planned()], tmp_path, RUN)
    actor = host.KubeControl(FakeAPI(), PREFIX, [planned()], tmp_path, RUN)
    with pytest.raises(ValueError, match="outside"):
        actor.get(planned("ClusterRole", "unrelated"))


def test_stale_controller_generation_cannot_be_ready():
    value = {"kind": "Deployment", "metadata": {"generation": 2}, "spec": {"replicas": 1},
             "status": {"observedGeneration": 1, "replicas": 1, "updatedReplicas": 1, "readyReplicas": 1, "availableReplicas": 1}}
    assert not host.controller_ready(value)
    value["status"]["observedGeneration"] = 2
    assert host.controller_ready(value)


def flagd_application():
    sidecar = {"name": "flagd-ui", "image": "ghcr.io/open-telemetry/demo@sha256:0738a32cecf129842d99843f470cfdb4091b9ff5b252d70b9850d5e24e82b438",
               "resources": {"limits": {"memory": "250Mi"}}}
    return {"items": [{"kind": "Deployment", "metadata": {"name": "flagd"},
                       "spec": {"template": {"spec": {"containers": [{"name": "flagd", "image": "unchanged"}, sidecar]}}}}]}


def test_editor_correction_preserves_image_memory_and_other_containers():
    app = flagd_application()
    original = copy.deepcopy(app)
    change = host.limit_flagd_files(app)
    sidecar = app["items"][0]["spec"]["template"]["spec"]["containers"][1]
    assert sidecar.pop("command") == ["sh", "-c", "ulimit -n 65536 && exec /app/bin/server"]
    assert app == original
    assert change["before"] is None and change["container"] == "flagd-ui"


@pytest.mark.parametrize("damage", ["image", "command", "args", "memory", "missing", "duplicate"])
def test_editor_correction_refuses_unchecked_startup_profiles(damage):
    app = flagd_application()
    containers = app["items"][0]["spec"]["template"]["spec"]["containers"]
    sidecar = containers[1]
    if damage in ("image", "command", "args"):
        sidecar[damage] = "different" if damage == "image" else []
    elif damage == "memory":
        sidecar["resources"]["limits"]["memory"] = "1Gi"
    elif damage == "missing":
        containers.pop()
    else:
        containers.append(copy.deepcopy(sidecar))
    original = copy.deepcopy(app)
    with pytest.raises(ValueError):
        host.limit_flagd_files(app)
    assert app == original


def topology():
    template = {"metadata": {"labels": {"app": "frontend"}}, "spec": {"containers": [{"name": "frontend", "image": "pinned"}]}}
    deployment = {"metadata": {"name": "frontend", "uid": "deployment"}, "spec": {"template": copy.deepcopy(template)}}
    rs_template = copy.deepcopy(template)
    rs_template["metadata"]["labels"]["pod-template-hash"] = "current"
    rs = {"metadata": {"uid": "replicaset", "ownerReferences": [{"kind": "Deployment", "uid": "deployment", "controller": True}]}, "spec": {"template": rs_template}}
    pod = {"metadata": {"uid": "pod", "labels": {"app": "frontend"}, "ownerReferences": [{"kind": "ReplicaSet", "uid": "replicaset", "controller": True}]},
           "status": {"podIP": "10.0.0.1", "conditions": [{"type": "Ready", "status": "True"}]}}
    service = {"metadata": {"name": "frontend", "namespace": NAMESPACE, "uid": "service"}, "spec": {"selector": {"app": "frontend"}}}
    endpoint = {"metadata": {"labels": {"kubernetes.io/service-name": "frontend"}, "ownerReferences": [{"kind": "Service", "uid": "service"}]},
                "endpoints": [{"conditions": {"ready": True}, "targetRef": {"kind": "Pod", "namespace": NAMESPACE, "uid": "pod"}, "addresses": ["10.0.0.1"]}]}
    return service, [endpoint], [pod], [rs], [deployment]


def test_workflow_routing_resolves_to_current_deployment_template():
    host.validate_routing(*topology())


@pytest.mark.parametrize("damage", ["address", "pod_uid", "service_uid", "selector", "unready", "stale_template", "wrong_controller"])
def test_ready_labels_cannot_hide_stale_or_wrong_routing(damage):
    service, slices, pods, replicasets, deployments = topology()
    endpoint = slices[0]["endpoints"][0]
    if damage == "address":
        endpoint["addresses"] = ["10.0.0.99"]
    elif damage == "pod_uid":
        endpoint["targetRef"]["uid"] = "old-pod"
    elif damage == "service_uid":
        slices[0]["metadata"]["ownerReferences"][0]["uid"] = "old-service"
    elif damage == "selector":
        service["spec"]["selector"] = {"app": "other"}
    elif damage == "unready":
        endpoint["conditions"]["ready"] = False
    elif damage == "stale_template":
        replicasets[0]["spec"]["template"]["spec"]["containers"][0]["image"] = "old-image"
    else:
        replicasets[0]["metadata"]["ownerReferences"][0]["uid"] = "different-deployment"
    with pytest.raises(ValueError):
        host.validate_routing(service, slices, pods, replicasets, deployments)


def test_runtime_image_identity_is_checked_beyond_pod_image_text():
    digest = "sha256:" + "a" * 64
    image = "example/app@" + digest
    pod = {"metadata": {}, "spec": {"containers": [{"name": "app", "image": image}]},
           "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}],
                      "containerStatuses": [{"name": "app", "imageID": "containerd://" + digest}]}}
    host.validate_pod(pod, {image: {digest}})
    pod["status"]["containerStatuses"][0]["imageID"] = "containerd://sha256:" + "b" * 64
    with pytest.raises(ValueError, match="image ID"):
        host.validate_pod(pod, {image: {digest}})


@pytest.mark.parametrize("damage", ["unready", "oom", "missing_service", "stale_controller"])
def test_application_health_is_rechecked_after_successful_requests(damage):
    digest = "sha256:" + "a" * 64
    image, images = "example/app@" + digest, {"example/app@" + digest: {digest}}
    data = {key: [] for key in ("deployments", "daemonsets", "pods", "services", "endpointslices", "replicasets")}
    for name in ("frontend-proxy", "frontend", "cart", "product-catalog", "valkey-cart"):
        service, slices, pods, replicasets, deployments = topology()
        service["metadata"].update(name=name, uid="service-" + name)
        slices[0]["metadata"]["labels"]["kubernetes.io/service-name"] = name
        slices[0]["metadata"]["ownerReferences"][0]["uid"] = "service-" + name
        slices[0]["endpoints"][0]["targetRef"]["uid"] = "pod-" + name
        pods[0]["metadata"]["uid"] = "pod-" + name
        pods[0]["metadata"]["ownerReferences"][0]["uid"] = "replicaset-" + name
        pods[0]["spec"] = {"containers": [{"name": "app", "image": image}]}
        pods[0]["status"].update(phase="Running", containerStatuses=[{"name": "app", "imageID": image}])
        replicasets[0]["metadata"]["uid"] = "replicaset-" + name
        replicasets[0]["metadata"]["ownerReferences"][0]["uid"] = "deployment-" + name
        deployment = deployments[0]
        deployment.update(kind="Deployment", status={"observedGeneration": 1, "replicas": 1, "updatedReplicas": 1, "readyReplicas": 1, "availableReplicas": 1})
        deployment["metadata"].update(name=name, uid="deployment-" + name, generation=1)
        deployment["spec"]["replicas"] = 1
        for key, values in (("services", [service]), ("endpointslices", slices), ("pods", pods), ("replicasets", replicasets), ("deployments", deployments)):
            data[key].extend(values)
    host.validate_application(data, images)
    if damage == "unready":
        next(c for c in data["pods"][0]["status"]["conditions"] if c["type"] == "Ready")["status"] = "False"
    elif damage == "oom":
        data["pods"][0]["status"]["containerStatuses"][0]["lastState"] = {"terminated": {"reason": "OOMKilled"}}
    elif damage == "missing_service":
        data["services"] = [s for s in data["services"] if s["metadata"]["name"] != "cart"]
    else:
        data["deployments"][0]["status"]["observedGeneration"] = 0
    with pytest.raises(ValueError):
        host.validate_application(data, images)


def test_independent_one_item_encoder_matches_literal_wire():
    assert host.expected_wire("bw-cart-u", "sku", 2) == "0a0962772d636172742d7512070a03736b751002"
    with pytest.raises(ValueError):
        host.expected_wire("bw-cart-u", "sku", True)


def logs_fixture():
    events, records = [], {}
    def snapshot(user):
        value = copy.deepcopy(records.get(user))
        return {"user": user, "key_type": "hash" if value else "none", "hash_fields": 1 if value else 0,
                "ttl_ms": 3599000 if value else -2, "cart": value,
                "raw_hex": host.expected_wire(user, "sku", value["items"][0]["quantity"]) if value else None}
    def send(method, path, body, request_id):
        if path.startswith("/api/products"):
            value = [copy.deepcopy(PRODUCT)]
        elif method == "POST":
            records[body["userId"]] = {"userId": body["userId"], "items": [copy.deepcopy(body["item"])]}
            value = copy.deepcopy(records[body["userId"]])
        else:
            user = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["sessionId"][0]
            value = copy.deepcopy(records[user])
            value["items"][0]["product"] = copy.deepcopy(PRODUCT)
        return {"status": 200, "body": value, "raw_body_b64": base64.b64encode(json.dumps(value).encode()).decode(),
                "request_id": request_id, "elapsed_s": .01, "deadline_s": 5}
    hashes = {"cart.py": "cart-hash", "probe.py": "probe-hash"}
    events.append({"stage": "manifest", "manifest": {"run_id": RUN, "namespace": NAMESPACE, "source_sha256": hashes, "readiness_only": False}})
    result = probe.exercise(send, snapshot, events.append, RUN, interval_s=0, settle_s=0)
    events.append({"stage": "summary", "summary": result})
    pod = {"metadata": {"ownerReferences": [{"kind": "Job", "uid": "job-uid", "controller": True}]},
           "status": {"phase": "Succeeded", "containerStatuses": [{"restartCount": 0, "state": {"terminated": {"exitCode": 0}}}],
                      "initContainerStatuses": [{"restartCount": 0, "state": {"terminated": {"exitCode": 0}}}]}}
    job = {"metadata": {"uid": "job-uid"}, "status": {"succeeded": 1}}
    return events, pod, job, hashes


def test_raw_control_audit_checks_network_bodies_and_store_bytes_without_project_decoder():
    events, pod, job, hashes = logs_fixture()
    result = host.audit_helper("\n".join(map(json.dumps, events)), pod, job, RUN, NAMESPACE, hashes)
    assert result == {"passed": True, "requests": 25, "write_attempts": 2, "independent_wire_checks": True}


@pytest.mark.parametrize("damage", ["fake_summary", "missing_initial", "missing_final", "lost_request", "raw_body", "stored_bytes", "expiry", "pod_failed", "retry", "source"])
def test_success_summary_cannot_hide_bad_evidence(damage):
    events, pod, job, hashes = logs_fixture()
    if damage == "fake_summary":
        events = [e for e in events if e["stage"] != "store"]
    elif damage == "missing_initial":
        events = [e for e in events if not (e["stage"] == "store" and e["snapshot"]["key_type"] == "none")]
    elif damage == "missing_final":
        last_response = max(i for i,e in enumerate(events) if e["stage"] == "response")
        events = [e for i,e in enumerate(events) if not (i > last_response and e["stage"] == "store")]
    elif damage == "lost_request":
        events = [e for e in events if not (e["stage"] == "response" and e["request_id"].endswith("025"))]
    elif damage == "raw_body":
        next(e for e in events if e["stage"] == "response")["response"]["raw_body_b64"] = "e30="
    elif damage in ("stored_bytes", "expiry"):
        value = next(e["snapshot"] for e in events if e["stage"] == "store" and e["snapshot"]["key_type"] == "hash")
        value["raw_hex" if damage == "stored_bytes" else "ttl_ms"] = "00" if damage == "stored_bytes" else -1
    elif damage == "pod_failed":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    elif damage == "retry":
        pod["status"]["containerStatuses"][0]["restartCount"] = 1
    else:
        hashes = {"cart.py": "wrong", "probe.py": "wrong"}
    with pytest.raises(ValueError):
        host.audit_helper("\n".join(map(json.dumps, events)), pod, job, RUN, NAMESPACE, hashes)


def test_command_failures_keep_raw_stdout_stderr_and_receipt(tmp_path):
    command = runner.Commands(tmp_path, {})
    with pytest.raises(RuntimeError):
        command(["/usr/bin/false"])
    receipt = json.loads((tmp_path / "command-0001.json").read_text())
    assert receipt["returncode"] == 1
    assert (tmp_path / "command-0001.stdout").exists() and (tmp_path / "command-0001.stderr").exists()


def test_command_timeout_does_not_retry_mutation(tmp_path, monkeypatch):
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], 20, output=b"partial", stderr=b"timeout")
    monkeypatch.setattr(runner.subprocess, "run", execute)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.Commands(tmp_path, {})(["owned-mutation"])
    assert len(calls) == 1
    assert (tmp_path / "command-0001.stdout").read_bytes() == b"partial"
    assert "not automatically retried" in json.loads((tmp_path / "command-0001.json").read_text())["error"]


def test_current_mac_cannot_accidentally_execute_linux_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux AMD64"):
        runner.run(tmp_path / "missing-bundle", tmp_path / "output", "default", "host-id")
    assert not (tmp_path / "output").exists()
