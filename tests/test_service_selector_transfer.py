import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from service_operator import selector, selector_transfer as transfer, tools


def response(body, **changes):
    return {"status": 200, "body": body, "elapsed_s": .01, "deadline_s": 5, **changes}


def hotel_check():
    return {"application": "hotel", "responses": [response({"type": "FeatureCollection", "features": [{"type": "Feature", "id": "4"}]}),
                                                 response({"message": "Reserve successfully!"})]}


def shop_check():
    product = {"id": "sku", "name": "Telescope", "picture": "/p.png", "description": "Telescope",
               "categories": [], "priceUsd": {"currencyCode": "USD", "units": 5, "nanos": 0}}
    item = {"productId": "sku", "quantity": 1}
    return {"application": "shop", "intent": {"userId": "bw-cart-check", "item": copy.deepcopy(item)},
            "responses": [response([product]), response({"userId": "bw-cart-check", "items": [copy.deepcopy(item)]}),
                          response({"userId": "bw-cart-check", "items": [{**item, "product": product}]})]}


@pytest.mark.parametrize("application", ["hotel", "shop"])
def test_profiles_share_actions_and_restrict_native_mutation_targets(application):
    profile = "selector_" + application
    context = transfer.public_task(application)
    definitions = {row["name"]: row for row in context["tools"]}
    assert set(definitions) == {"inspect", "logs", "set_selector", "restart", "check", "wait", "finish", "clock"}
    assert definitions["set_selector"]["parameters"]["properties"]["service"]["enum"] == ["frontend"]
    assert all(condition not in json.dumps(context) for condition in transfer.CONDITIONS)
    rules = tools.operator_role("app", profile)[1]["rules"]
    writes = [rule for rule in rules if "patch" in rule["verbs"]]
    assert len(writes) == 2 and all(rule["resourceNames"] == ["frontend"] for rule in writes)
    assert not any("secrets" in rule["resources"] or "pods/exec" in rule["resources"] for rule in rules)
    with pytest.raises(ValueError, match="deadline"):
        tools.PublicTools(None, None, None, time.monotonic(), profile=profile)


@pytest.mark.parametrize("application,target", [("hotel", "mongodb-reservation"), ("shop", "valkey-cart"), ("shop", "postgresql")])
@pytest.mark.parametrize("operation", ["set_selector", "restart"])
def test_stateful_mutations_fail_at_schema_and_backend(application, target, operation):
    calls = []
    backend = transfer.SelectorBackend(SimpleNamespace(kube=lambda *args, **kwargs: calls.append(args)), "app", application)
    args = {"service": target, "selector": {"app": "other"}} if operation == "set_selector" else {"deployment": target}
    with pytest.raises(tools.ToolInputError):
        backend.perform(operation, args)
    assert calls == []


def deployment(label="opentelemetry.io/name"):
    return {"metadata": {"uid": "deployment", "resourceVersion": "7", "generation": 3},
            "spec": {"replicas": 1, "selector": {"matchLabels": {label: "frontend"}},
                     "template": {"metadata": {"labels": {label: "frontend"}, "annotations": {"keep": "value"}}}},
            "status": {"observedGeneration": 3, "replicas": 1, "updatedReplicas": 1, "readyReplicas": 1, "availableReplicas": 1}}


@pytest.mark.parametrize("label,application", [("io.kompose.service", "hotel"), ("opentelemetry.io/name", "shop")])
def test_pod_lookup_uses_observed_deployment_selector_and_actor_identity(label, application):
    commands = []
    def kube(args, **kwargs):
        commands.append(args)
        return json.dumps(deployment(label) if "deployment" in args else {"items": []})
    backend = transfer.SelectorBackend(SimpleNamespace(kube=kube), "owned", application)
    assert backend.perform("inspect", {"resource": "pods", "name": "frontend"}) == {"pods": []}
    assert all("--as=system:serviceaccount:owned:operator" in args and args[args.index("-n") + 1] == "owned" for args in commands)
    assert commands[-1][commands[-1].index("-l") + 1] == label + "=frontend"


def test_set_based_selectors_keep_all_kubernetes_requirements():
    value = transfer.deployment_selector({"matchLabels": {"app": "frontend"}, "matchExpressions": [
        {"key": "tier", "operator": "In", "values": ["web", "api"]},
        {"key": "stage", "operator": "NotIn", "values": ["dev"]},
        {"key": "live", "operator": "Exists"}, {"key": "retired", "operator": "DoesNotExist"}]})
    assert value == "app=frontend,tier in (api,web),stage notin (dev),live,!retired"
    with pytest.raises(ValueError):
        transfer.deployment_selector({"matchExpressions": [{"key": "app", "operator": "Unsupported"}]})


@pytest.mark.parametrize("operation", ["set_selector", "restart"])
def test_mutation_uses_atomic_identity_tests_and_keeps_unrelated_fields(operation):
    commands = []
    def kube(args, **kwargs):
        commands.append(args)
        if "get" in args:
            return json.dumps({"metadata": {"uid": "service", "resourceVersion": "9"}} if "service" in args else deployment())
        patch = json.loads(args[-1])
        assert [row["op"] for row in patch] == ["test", "test", "replace" if operation == "set_selector" else "add"]
        assert patch[:2] == [{"op": "test", "path": "/metadata/uid", "value": "service" if operation == "set_selector" else "deployment"},
                             {"op": "test", "path": "/metadata/resourceVersion", "value": "9" if operation == "set_selector" else "7"}]
        if operation == "restart":
            assert patch[-1]["value"]["keep"] == "value"
        raise RuntimeError("API rejected stale resource version")
    backend = transfer.SelectorBackend(SimpleNamespace(kube=kube), "owned", "shop")
    args = {"service": "frontend", "selector": {"app": "frontend"}} if operation == "set_selector" else {"deployment": "frontend"}
    with pytest.raises(RuntimeError, match="stale"):
        backend.perform(operation, args)
    assert len(commands) == 2


@pytest.mark.parametrize("check", [hotel_check(), shop_check()])
def test_public_workflow_checks_meaningful_responses_without_private_grade(check):
    assert transfer.public_workflow_valid(check)
    assert "grade" not in check and "state_preserved" not in check


@pytest.mark.parametrize("damage", ["empty_hotels", "fake_booking", "late", "wrong_user", "wrong_product", "duplicate_write", "wrong_readback"])
def test_http_200_cannot_replace_a_valid_public_workflow(damage):
    value = hotel_check() if damage in ("empty_hotels", "fake_booking") else shop_check()
    if damage == "empty_hotels":
        value["responses"][0]["body"]["features"] = []
    elif damage == "fake_booking":
        value["responses"][1]["body"] = {"message": "success"}
    elif damage == "late":
        value["responses"][1]["elapsed_s"] = 6
    elif damage == "wrong_user":
        value["responses"][1]["body"]["userId"] = "other"
    elif damage == "wrong_product":
        value["intent"]["item"]["productId"] = "absent"
    elif damage == "duplicate_write":
        value["responses"][1]["body"]["items"][0]["quantity"] = 2
        value["intent"]["item"]["quantity"] = 1
    else:
        value["responses"][2]["body"]["items"][0]["quantity"] = 2
    assert not transfer.public_workflow_valid(value)


class PublicObservations:
    def __init__(self, faulty, empty_as_null=False):
        self.faulty = faulty
        self.empty_as_null = empty_as_null
        self.calls = []

    def send(self, call):
        name, args = call["tool"], call["arguments"]
        self.calls.append(call)
        value = {}
        if name == "inspect":
            if args["resource"] == "deployment":
                value = {"generation": 3, "replicas": 1, "status": deployment()["status"], "pod_labels": {"opentelemetry.io/name": "frontend"}}
            elif args["resource"] == "pods":
                value = {"pods": [{"uid": "live-frontend", "deleting": False, "conditions": [{"type": "Ready", "status": "True"}]}]}
            elif args["resource"] == "endpoints":
                value = {"slices": [{"endpoints": (None if self.empty_as_null else []) if self.faulty else [{"conditions": {"ready": True}, "targetRef": {"uid": "live-frontend"}}]}]}
            else:
                value = {"selector": {"opentelemetry.io/name": "frontend", "mismatch": "extra"}}
        elif name == "set_selector":
            assert args["selector"] == {"opentelemetry.io/name": "frontend"}
            self.faulty = False
        elif name == "check":
            value = shop_check()
        return {"ok": True, "result": value}


@pytest.mark.parametrize("faulty,empty_as_null", [(False, False), (True, False), (True, True)])
def test_public_runbook_investigates_and_repairs_without_a_private_case(faulty, empty_as_null):
    source = PublicObservations(faulty, empty_as_null)
    transfer.investigate(source.send)
    names = [row["tool"] for row in source.calls]
    assert names[-1] == "finish" and names.count("check") == 1
    assert names.count("set_selector") == int(faulty)
    assert "restart" not in names


def test_whole_attempt_loss_counts_missing_late_undelivered_and_invalid_work():
    offers = [{"request_id": str(i), "offered_s": i, "deadline_s": 1} for i in range(5)]
    outcomes = [{"request_id": str(i), "finished_s": i + (.2 if i != 1 else 2), "valid": i != 2, "delivered": i != 3} for i in range(4)]
    loss = transfer.workload_loss(offers, outcomes, 0, 6)
    assert loss == {"offered": 5, "completed": 4, "valid": 1, "failed": 4, "missing_outcomes": 1,
                    "opened_s": 0, "ended_s": 6, "duration_s": 6}


@pytest.mark.parametrize("damage", ["duplicate_offer", "duplicate_outcome", "unknown_outcome", "bad_clock", "nan_horizon"])
def test_loss_accounting_refuses_inconsistent_journals(damage):
    offers = [{"request_id": "one", "offered_s": 1, "deadline_s": 1}]
    outcomes = [{"request_id": "one", "finished_s": 1.5, "valid": True, "delivered": True}]
    if damage == "duplicate_offer":
        offers *= 2
    elif damage == "duplicate_outcome":
        outcomes *= 2
    elif damage == "unknown_outcome":
        outcomes[0]["request_id"] = "missing"
    elif damage == "bad_clock":
        outcomes[0]["finished_s"] = 0
    with pytest.raises(ValueError):
        transfer.workload_loss(offers, outcomes, 0, float("nan") if damage == "nan_horizon" else 4)


def test_loss_counts_completion_after_the_horizon_as_failed_work():
    offers = [{"request_id": "one", "offered_s": 1, "deadline_s": 5}]
    outcomes = [{"request_id": "one", "finished_s": 3, "valid": True, "delivered": True}]
    result = transfer.workload_loss(offers, outcomes, 0, 2)
    assert result["valid"] == 0 and result["failed"] == 1 and result["completed"] == 1


@pytest.mark.parametrize("field", ["workflow_valid", "state_preserved", "route_valid", "application_healthy", "recovered"])
def test_finish_cannot_hide_private_application_or_state_failures(field):
    values = {"workflow_valid": True, "state_preserved": True, "route_valid": True, "application_healthy": True,
              "recovered": True, "post_completion": 2}
    public = SimpleNamespace(started=0, finished_at=1, deadline_s=300, mutation_attempts=0)
    assert transfer.grade("known_repair", public, **values)["success"]
    values[field] = False
    assert not transfer.grade("premature_finish", public, **values)["success"]


def test_healthy_mutation_and_no_post_finish_work_are_rejected():
    values = {"workflow_valid": True, "state_preserved": True, "route_valid": True, "application_healthy": True,
              "recovered": True, "post_completion": 2}
    public = SimpleNamespace(started=0, finished_at=1, deadline_s=300, mutation_attempts=1)
    assert not transfer.grade("healthy", public, **values)["success"]
    public.mutation_attempts = 0
    assert not transfer.grade("healthy", public, **{**values, "post_completion": 0})["success"]


def test_preparer_keeps_injection_and_private_answers_out_of_public_packet(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "experiments/service_selector_transfer.py"
    spec = importlib.util.spec_from_file_location("selector_preparer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    injector = tmp_path / "injector.py"
    body = b'class VirtualizationFaultInjector:\n def inject_wrong_service_selector(self, microservices): return microservices\n def recover_wrong_service_selector(self, microservices): return microservices\n'
    injector.write_bytes(body)
    monkeypatch.setattr(selector, "INJECTOR_BLOB", hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest())
    objects = tmp_path / "objects.json"
    dep = deployment();dep.update(kind="Deployment");dep["metadata"]["name"] = "frontend"
    objects.write_text(json.dumps({"items": [dep, {"kind": "Service", "metadata": {"name": "frontend"}, "spec": {"selector": {"opentelemetry.io/name": "frontend"}}}]}))
    output = tmp_path / "prepared"
    manifest = module.prepare("shop", objects, injector, "bw-astro-0123456789", output)
    public = json.loads((output / "public-task.json").read_text())
    assert not any(text in json.dumps(public) for text in ("current_service_name", "inject_wrong", "expected_grade", "original_selector"))
    assert manifest["paid_execution"] is False and manifest["native_ready"] is False
    private = json.loads((output / "private-control-plan.json").read_text())
    assert [row["expected_grade"] for row in private["conditions"]] == [True, True, False, False]
    assert private["injected_selector"]["current_service_name"] == "frontend"
    with pytest.raises(FileExistsError):
        module.prepare("shop", objects, injector, "bw-astro-0123456789", output)
