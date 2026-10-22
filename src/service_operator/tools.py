import json
import math
import re
import time
from datetime import datetime, timezone


APPS = ("frontend", "search", "user", "reservation", "consul", "jaeger",
        "memcached-reserve", "mongodb-user", "mongodb-reservation")
RETRY_APPS = APPS + ("geo", "rate", "profile", "mongodb-geo", "mongodb-rate", "mongodb-profile",
                     "memcached-rate", "memcached-profile")
POLICY_FIELDS = {
    "configure_rate": {"qps_limit": ("RATE_BACKEND_QPS_LIMIT", "integer", 1, 500),
                       "queue_capacity": ("RATE_QUEUE_CAPACITY", "integer", 1, 256)},
    "configure_search": {"timeout_ms": ("RATE_RPC_TIMEOUT_MS", "integer", 1, 5000),
                         "max_attempts": ("RATE_RPC_MAX_ATTEMPTS", "integer", 1, 8),
                         "backoff_ms": ("RATE_RPC_INITIAL_BACKOFF_MS", "integer", 1, 1000),
                         "backoff_multiplier": ("RATE_RPC_BACKOFF_MULTIPLIER", "number", 1, 4),
                         "jitter": ("RATE_RPC_JITTER", "number", 0, 1)},
}
LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?\Z")
PREFIX = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
TOOL_SPEC = {
    "inspect": {"required": ["resource", "name"], "resources": ["service", "deployment", "pods", "endpoints", "events"]},
    "logs": {"required": ["deployment", "lines"], "lines": [1, 40]},
    "set_selector": {"required": ["service", "selector"], "max_labels": 8},
    "restart": {"required": ["deployment"]},
    "check": {"required": [], "max_calls": 2},
    "wait": {"required": ["seconds"], "seconds": [1, 3]},
    "finish": {"required": ["summary"], "max_summary_chars": 256},
}


class ToolInputError(ValueError):
    pass


def contract(profile):
    if profile == "booking":
        return APPS, TOOL_SPEC
    if profile not in ("retry", "pilot"):
        raise ValueError("unknown tool profile")
    specs = {name: spec for name, spec in TOOL_SPEC.items() if name != "set_selector" or profile == "pilot"}
    specs["wait"] = {"required": ["seconds"], "seconds": [1, 10]}
    specs.update({name: {"required": list(fields)} for name, fields in POLICY_FIELDS.items()})
    specs.update({"metrics": {"required": ["service"]}, "clock": {"required": []}})
    return RETRY_APPS, specs


def tool_definitions(profile="booking"):
    apps, specs = contract(profile)
    properties = {
        "inspect": {"resource": {"type": "string", "enum": TOOL_SPEC["inspect"]["resources"]},
                    "name": {"type": "string", "enum": list(apps)}},
        "logs": {"deployment": {"type": "string", "enum": list(apps)},
                 "lines": {"type": "integer", "minimum": 1, "maximum": 40}},
        "set_selector": {"service": {"type": "string", "enum": list(apps)},
                         "selector": {"type": "object", "minProperties": 1, "maxProperties": 8,
                                      "additionalProperties": {"type": "string"}}},
        "restart": {"deployment": {"type": "string", "enum": list(apps)}},
        "check": {}, "wait": {"seconds": {"type": "integer", "minimum": 1, "maximum": specs["wait"]["seconds"][1]}},
        "finish": {"summary": {"type": "string", "maxLength": 256}},
    }
    descriptions = {"inspect": "Inspect one application resource.", "logs": "Read bounded application logs.",
                    "set_selector": "Replace a Service selector.", "restart": "Request a deployment restart.",
                    "check": "Try a fresh booking and report its HTTP response.", "wait": "Wait for an operation to take effect.",
                    "finish": "Declare the investigation complete."}
    if profile in ("retry", "pilot"):
        properties["restart"]["deployment"]["enum"] = ["frontend", "search", "rate"]
        if profile == "pilot":
            properties["set_selector"]["service"]["enum"] = ["frontend", "search", "rate"]
        properties.update({"metrics": {"service": {"type": "string", "enum": ["search", "rate"]}}, "clock": {}})
        descriptions.update({"metrics": "Read current application counters and gauges; counters may reset after rollout.",
                             "clock": "Read elapsed time and remaining action/check budgets.",
                             "check": "Search for hotels, then try one booking if a candidate is returned; report HTTP responses."})
        for name, fields in POLICY_FIELDS.items():
            properties[name] = {key: {"type": value[1], "minimum": value[2], "maximum": value[3]}
                                for key, value in fields.items()}
            descriptions[name] = "Request a deployment rollout with these policy fields; inspect rollout and runtime behavior afterward."
    return [{"name": name, "description": descriptions[name], "parameters": {
        "type": "object", "properties": properties[name], "required": spec["required"], "additionalProperties": False}}
        for name, spec in specs.items()]


def validate_call(call, profile="booking"):
    apps, specs = contract(profile)
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        raise ToolInputError("expected tool and arguments only")
    name, args = call["tool"], call["arguments"]
    if not isinstance(name, str) or name not in specs or not isinstance(args, dict):
        raise ToolInputError("unknown tool or invalid arguments")
    if set(args) != set(specs[name]["required"]):
        raise ToolInputError("arguments do not match the tool contract")
    for key in ("name", "service", "deployment"):
        if key in args and (not isinstance(args[key], str) or args[key] not in apps):
            raise ToolInputError("resource is not available")
    if name == "inspect" and args["resource"] not in specs[name]["resources"]:
        raise ToolInputError("resource type is not available")
    if name in ("logs", "wait"):
        key, low, high = ("lines", 1, 40) if name == "logs" else ("seconds", *specs["wait"]["seconds"])
        if type(args[key]) is not int or not low <= args[key] <= high:
            raise ToolInputError("argument is outside its allowed range")
    if profile in ("retry", "pilot"):
        if name == "set_selector" and args["service"] not in ("frontend", "search", "rate"):
            raise ToolInputError("selector target is not available")
        if name == "metrics" and args["service"] not in ("search", "rate"):
            raise ToolInputError("metrics service is not available")
        if name == "restart" and args["deployment"] not in ("frontend", "search", "rate"):
            raise ToolInputError("restart target is not available")
        if name in POLICY_FIELDS:
            for key, (_, kind, low, high) in POLICY_FIELDS[name].items():
                value = args[key]
                allowed = (int,) if kind == "integer" else (int, float)
                if type(value) not in allowed or not math.isfinite(value) or not low <= value <= high:
                    raise ToolInputError("policy field is outside its allowed range")
    if name == "finish" and (not isinstance(args["summary"], str) or len(args["summary"]) > 256):
        raise ToolInputError("completion summary is too long or invalid")
    if name == "set_selector":
        labels = args["selector"]
        if not isinstance(labels, dict) or not 1 <= len(labels) <= 8:
            raise ToolInputError("selector needs one to eight labels")
        for key, value in labels.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ToolInputError("label keys and values must be strings")
            parts = key.split("/")
            if (len(parts) > 2 or not LABEL.fullmatch(parts[-1])
                    or len(parts) == 2 and (len(parts[0]) > 253 or not all(PREFIX.fullmatch(p) for p in parts[0].split(".")))
                    or value and not LABEL.fullmatch(value)):
                raise ToolInputError("invalid label syntax")
    return name, args


class PublicTools:
    def __init__(self, backend, check, record, started, max_calls=16, complete=None,
                 profile="booking", deadline_s=None):
        contract(profile)
        if deadline_s is not None and (type(deadline_s) not in (int, float)
                                      or not math.isfinite(deadline_s) or deadline_s <= 0):
            raise ValueError("episode deadline must be positive and finite")
        if profile in ("retry", "pilot") and deadline_s is None:
            raise ValueError("timed episodes require a public deadline")
        self.backend, self.check, self.record = backend, check, record
        self.started, self.max_calls = started, max_calls
        self.calls = 0
        self.checks = 0
        self.mutations = 0
        self.mutation_attempts = 0
        self.finished_at = None
        self.complete = complete
        self.profile, self.deadline_s = profile, deadline_s

    def clock(self):
        elapsed = max(0, time.monotonic() - self.started)
        return {"elapsed_s": round(elapsed, 6), "remaining_s": round(max(0, self.deadline_s - elapsed), 6),
                "action_limit_s": self.deadline_s, "remaining_calls": max(0, self.max_calls - self.calls),
                "remaining_checks": max(0, 2 - self.checks),
                "closed": self.finished_at is not None or elapsed >= self.deadline_s or self.calls >= self.max_calls}

    def invoke(self, call):
        self.calls += 1
        event = {"call": call, "started_s": time.monotonic() - self.started}
        try:
            if self.calls > self.max_calls:
                raise ToolInputError("tool call budget exhausted")
            if self.finished_at is not None:
                raise ToolInputError("episode already declared complete")
            if self.deadline_s is not None and time.monotonic() - self.started >= self.deadline_s:
                raise ToolInputError("episode action deadline reached")
            name, args = validate_call(call, self.profile)
            if name == "finish":
                if self.complete is not None:
                    self.complete()
                self.finished_at = time.monotonic()
                response = {"declared_complete": True}
            elif name == "wait":
                time.sleep(args["seconds"])
                response = {"waited_s": args["seconds"]}
            elif name == "check":
                if self.checks >= 2:
                    raise ToolInputError("functional check budget exhausted")
                self.checks += 1
                response = self.check()
            elif name == "clock":
                response = self.clock()
            else:
                if name in ("set_selector", "restart", *POLICY_FIELDS):
                    self.mutation_attempts += 1
                response = self.backend.perform(name, args)
                if name in ("set_selector", "restart", *POLICY_FIELDS):
                    self.mutations += 1
            result = {"ok": True, "result": response}
        except ToolInputError as exc:
            result = {"ok": False, "error": {"code": "invalid_request", "message": str(exc)}}
        except Exception:
            result = {"ok": False, "error": {"code": "operation_failed", "message": "operation failed; inspect application state"}}
        if self.deadline_s is not None:
            result["clock"] = self.clock()
        event.update({"response": result, "finished_s": time.monotonic() - self.started})
        self.record(event)
        return result


def operator_role(namespace, profile="booking"):
    contract(profile)
    objects = [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "operator", "namespace": namespace},
         "automountServiceAccountToken": False},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role", "metadata": {"name": "operator", "namespace": namespace},
         "rules": [
             {"apiGroups": [""], "resources": ["services"], "verbs": ["get", "list", "patch"]},
             {"apiGroups": [""], "resources": ["pods", "events"], "verbs": ["get", "list"]},
             {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
             {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["get", "list", "patch"]},
             {"apiGroups": ["apps"], "resources": ["replicasets"], "verbs": ["get", "list"]},
             {"apiGroups": ["discovery.k8s.io"], "resources": ["endpointslices"], "verbs": ["get", "list"]},
         ]},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding",
         "metadata": {"name": "operator", "namespace": namespace},
         "subjects": [{"kind": "ServiceAccount", "name": "operator", "namespace": namespace}],
         "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "operator"}},
    ]
    if profile in ("retry", "pilot"):
        rules = objects[1]["rules"]
        rules[0]["verbs"] = ["get", "list"]
        rules[3]["verbs"] = ["get", "list"]
        rules += [{"apiGroups": [""], "resources": ["services/proxy"], "verbs": ["get"],
                   "resourceNames": ["search-metrics", "rate-metrics"]},
                  {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["patch"],
                   "resourceNames": ["frontend", "search", "rate"]}]
        if profile == "pilot":
            rules += [{"apiGroups": [""], "resources": ["services"], "verbs": ["patch"],
                       "resourceNames": ["frontend", "search", "rate"]}]
    return objects


class OperatorBackend:
    def __init__(self, runtime, namespace):
        self.runtime, self.namespace = runtime, namespace
        self.identity = ["--as=system:serviceaccount:" + namespace + ":operator",
                         "--as-group=system:authenticated", "--as-group=system:serviceaccounts",
                         "--as-group=system:serviceaccounts:" + namespace]

    def kube(self, args, **kwargs):
        return self.runtime.kube([*self.identity, "-n", self.namespace, *args], **kwargs)

    def perform(self, tool, args):
        if tool == "set_selector":
            patch = [{"op": "replace", "path": "/spec/selector", "value": args["selector"]}]
            self.kube(["patch", "service", args["service"], "--type=json", "-p", json.dumps(patch)])
            return {"service": args["service"], "selector": args["selector"]}
        if tool == "restart":
            now = datetime.now(timezone.utc).isoformat()
            patch = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}}
            self.kube(["patch", "deployment", args["deployment"], "--type=merge", "-p", json.dumps(patch)])
            return {"deployment": args["deployment"], "restart_requested": True}
        if tool == "logs":
            return {"deployment": args["deployment"], "text": self.kube([
                "logs", "deployment/" + args["deployment"], "--tail=" + str(args["lines"]), "--limit-bytes=8192"])}
        resource, name = args["resource"], args["name"]
        if resource in ("service", "deployment"):
            obj = json.loads(self.kube(["get", resource, name, "-o", "json"]))
            if resource == "service":
                return {"name": name, "selector": obj["spec"].get("selector", {}), "ports": obj["spec"]["ports"]}
            return {"name": name, "replicas": obj["spec"].get("replicas"), "status": obj.get("status", {}),
                    "pod_labels": obj["spec"]["template"]["metadata"]["labels"]}
        if resource == "pods":
            rows = json.loads(self.kube(["get", "pods", "-l", "io.kompose.service=" + name, "-o", "json"]))["items"]
            return {"pods": [{"name": row["metadata"]["name"], "labels": row["metadata"]["labels"],
                               "conditions": row.get("status", {}).get("conditions", []),
                               "containers": [{"name": c["name"], "command": c.get("command"),
                                               "readiness": c.get("readinessProbe")} for c in row["spec"]["containers"]]}
                              for row in rows]}
        if resource == "endpoints":
            rows = json.loads(self.kube(["get", "endpointslices", "-l", "kubernetes.io/service-name=" + name, "-o", "json"]))["items"]
            return {"service": name, "slices": [{"ports": row.get("ports", []), "endpoints": row.get("endpoints", [])} for row in rows]}
        rows = json.loads(self.kube(["get", "events", "-o", "json"]))["items"]
        return {"events": [{"reason": row.get("reason"), "message": row.get("message", "")[:400]}
                           for row in rows if row.get("involvedObject", {}).get("name", "").startswith(name)][-20:]}
