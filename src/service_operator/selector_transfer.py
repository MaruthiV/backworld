import json
import math
from datetime import datetime, timezone

from service_operator import cart, tools, workflow


CONDITIONS = ("healthy", "known_repair", "restart_only", "premature_finish")


def public_task(application):
    profile = {"hotel": "selector_hotel", "shop": "selector_shop"}[application]
    operation = "search for a hotel and create one booking" if application == "hotel" else "read the catalog and add one item to a cart"
    return {"task": f"Investigate this application workflow alert. Preserve existing user state, {operation}, and decide whether changes are needed.",
            "application": application, "tools": tools.tool_definitions(profile),
            "action_limit_s": 300, "max_tool_calls": 64, "max_checks": 2,
            "completion": "declares completion; protected traffic and private evaluation continue"}


def public_workflow_valid(check):
    responses = check.get("responses")
    if not isinstance(responses, list) or not responses or any(not isinstance(row, dict) or not cart.timely(row)
                                                             or row["deadline_s"] <= 0 for row in responses):
        return False
    if check.get("application") == "hotel" and len(responses) == 2:
        body = responses[0].get("body")
        return (isinstance(body, dict) and body.get("type") == "FeatureCollection"
                and isinstance(body.get("features"), list) and bool(body["features"])
                and all(isinstance(row, dict) and row.get("type") == "Feature" and isinstance(row.get("id"), str) and row["id"]
                        for row in body["features"])
                and len({row["id"] for row in body["features"]}) == len(body["features"])
                and responses[1].get("body") == workflow.BOOKING_MESSAGE)
    if check.get("application") == "shop" and len(responses) == 3:
        products, intent = responses[0].get("body"), check.get("intent")
        if (not isinstance(products, list) or not products or any(not cart.valid_product(row) for row in products)
                or len({row["id"] for row in products}) != len(products)
                or not isinstance(intent, dict) or set(intent) != {"userId", "item"}
                or not isinstance(intent["userId"], str) or not intent["userId"].startswith("bw-cart-")
                or not isinstance(intent["item"], dict) or set(intent["item"]) != {"productId", "quantity"}
                or intent["item"]["productId"] not in {row["id"] for row in products}
                or type(intent["item"]["quantity"]) is not int or intent["item"]["quantity"] != 1):
            return False
        expected = {"userId": intent["userId"], "items": [intent["item"]]}
        return cart.matches(responses[1], expected) and cart.matches(responses[2], expected, enriched=True)
    return False


def deployment_selector(selector):
    parts = []
    for key, value in sorted(selector.get("matchLabels", {}).items()):
        tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": {key: value}}}, "selector_shop")
        parts.append(key + "=" + value)
    for expression in selector.get("matchExpressions", []):
        key, operator, values = expression["key"], expression["operator"], expression.get("values", [])
        tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": {key: ""}}}, "selector_shop")
        if operator in ("In", "NotIn") and isinstance(values, list) and values:
            for value in values:
                tools.validate_call({"tool": "set_selector", "arguments": {"service": "frontend", "selector": {key: value}}}, "selector_shop")
            parts.append(key + (" in (" if operator == "In" else " notin (") + ",".join(sorted(values)) + ")")
        elif operator in ("Exists", "DoesNotExist") and values == []:
            parts.append(("!" if operator == "DoesNotExist" else "") + key)
        else:
            raise ValueError("unsupported or invalid Deployment selector")
    if not parts or set(selector) - {"matchLabels", "matchExpressions"}:
        raise ValueError("Deployment selector is empty or has unknown fields")
    return ",".join(parts)


class SelectorBackend(tools.OperatorBackend):
    def __init__(self, runtime, namespace, application):
        self.profile = {"hotel": "selector_hotel", "shop": "selector_shop"}[application]
        super().__init__(runtime, namespace)

    def perform(self, tool, args):
        tools.validate_call({"tool": tool, "arguments": args}, self.profile)
        if tool == "set_selector":
            obj = json.loads(self.kube(["get", "service", args["service"], "-o", "json"]))
            patch = [{"op": "test", "path": "/metadata/uid", "value": obj["metadata"]["uid"]},
                     {"op": "test", "path": "/metadata/resourceVersion", "value": obj["metadata"]["resourceVersion"]},
                     {"op": "replace", "path": "/spec/selector", "value": args["selector"]}]
            self.kube(["patch", "service", args["service"], "--type=json", "-p", json.dumps(patch)])
            return {"service": args["service"], "selector": args["selector"]}
        if tool == "restart":
            obj = json.loads(self.kube(["get", "deployment", args["deployment"], "-o", "json"]))
            annotations = {**obj["spec"]["template"]["metadata"].get("annotations", {}),
                           "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()}
            patch = [{"op": "test", "path": "/metadata/uid", "value": obj["metadata"]["uid"]},
                     {"op": "test", "path": "/metadata/resourceVersion", "value": obj["metadata"]["resourceVersion"]},
                     {"op": "add", "path": "/spec/template/metadata/annotations", "value": annotations}]
            self.kube(["patch", "deployment", args["deployment"], "--type=json", "-p", json.dumps(patch)])
            return {"deployment": args["deployment"], "restart_requested": True}
        if tool == "inspect" and args["resource"] in ("pods", "deployment"):
            obj = json.loads(self.kube(["get", "deployment", args["name"], "-o", "json"]))
            selector = deployment_selector(obj["spec"]["selector"])
            if args["resource"] == "deployment":
                return {"name": args["name"], "uid": obj["metadata"]["uid"], "generation": obj["metadata"]["generation"],
                        "replicas": obj["spec"].get("replicas", 1), "status": obj.get("status", {}),
                        "pod_labels": obj["spec"]["template"]["metadata"]["labels"], "selector": obj["spec"]["selector"]}
            rows = json.loads(self.kube(["get", "pods", "-l", selector, "-o", "json"]))["items"]
            return {"pods": [{"name": pod["metadata"]["name"], "uid": pod["metadata"]["uid"],
                              "labels": pod["metadata"].get("labels", {}),
                              "deleting": pod["metadata"].get("deletionTimestamp") is not None,
                              "conditions": pod.get("status", {}).get("conditions", []),
                              "owners": pod["metadata"].get("ownerReferences", [])} for pod in rows]}
        return super().perform(tool, args)


def investigate(send):
    def call(tool, **args):
        response = send({"tool": tool, "arguments": args})
        if not response["ok"]:
            raise RuntimeError("public selector operation failed: " + tool)
        return response["result"]

    def endpoints():
        data = call("inspect", resource="endpoints", name="frontend")
        return {endpoint.get("targetRef", {}).get("uid") for row in data["slices"] for endpoint in (row.get("endpoints") or [])
                if endpoint.get("conditions", {}).get("ready") is True}

    call("clock")
    deployment = call("inspect", resource="deployment", name="frontend")
    status = deployment["status"]
    if (status.get("observedGeneration", 0) < deployment["generation"]
            or not deployment["replicas"] or any(status.get(key) != deployment["replicas"]
                                                  for key in ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas"))):
        raise RuntimeError("frontend rollout is not current and ready")
    pods = call("inspect", resource="pods", name="frontend")["pods"]
    ready = {pod["uid"] for pod in pods if not pod["deleting"] and any(
        item.get("type") == "Ready" and item.get("status") == "True" for item in pod["conditions"])}
    if not ready:
        raise RuntimeError("no ready frontend Pods")
    if endpoints() != ready:
        service = call("inspect", resource="service", name="frontend")
        selector = deployment["pod_labels"]
        if service["selector"] == selector:
            raise RuntimeError("no justified selector change")
        call("set_selector", service="frontend", selector=selector)
        for _ in range(10):
            if endpoints() == ready:
                break
            call("wait", seconds=2)
        else:
            raise RuntimeError("frontend endpoints did not converge")
    check = call("check")
    if not public_workflow_valid(check):
        raise RuntimeError("application workflow check failed")
    call("finish", summary="frontend routing and public workflow checked")


def workload_loss(offers, outcomes, opened_s, ended_s):
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (opened_s, ended_s)) or ended_s <= opened_s:
        raise ValueError("invalid observation horizon")
    indexed = {}
    for offer in offers:
        identity, stamp, deadline = offer["request_id"], offer["offered_s"], offer["deadline_s"]
        if (not isinstance(identity, str) or not identity or identity in indexed
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in (stamp, deadline))
                or not opened_s <= stamp < ended_s or not 0 < deadline <= 5):
            raise ValueError("invalid or duplicate source offer")
        indexed[identity] = offer
    completed, valid = set(), 0
    for outcome in outcomes:
        identity, finished = outcome["request_id"], outcome["finished_s"]
        if identity not in indexed or identity in completed:
            raise ValueError("outcome has no unique source offer")
        offer = indexed[identity]
        if type(finished) not in (int, float) or not math.isfinite(finished) or finished < offer["offered_s"]:
            raise ValueError("invalid completion clock")
        completed.add(identity)
        valid += (outcome.get("valid") is True and outcome.get("delivered") is True
                  and finished <= min(ended_s, offer["offered_s"] + offer["deadline_s"]))
    return {"offered": len(indexed), "completed": len(completed), "valid": valid,
            "failed": len(indexed) - valid, "missing_outcomes": len(indexed) - len(completed),
            "opened_s": opened_s, "ended_s": ended_s, "duration_s": ended_s - opened_s}


def grade(condition, public, *, workflow_valid, state_preserved, route_valid, application_healthy, recovered, post_completion):
    if condition not in CONDITIONS:
        raise ValueError("unknown private control condition")
    on_time = public.finished_at is not None and 0 <= public.finished_at - public.started < public.deadline_s
    checks = {"declared_on_time": on_time, "workflow_valid": workflow_valid, "state_preserved": state_preserved,
              "route_valid": route_valid, "application_healthy": application_healthy, "recovered": recovered,
              "post_completion_work": type(post_completion) is int and post_completion >= 2,
              "healthy_no_mutations": condition != "healthy" or public.mutation_attempts == 0}
    return {"success": all(value is True for value in checks.values()), **checks}
