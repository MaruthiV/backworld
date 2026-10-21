import json
import re
import time
from datetime import datetime, timezone


APPS = ("frontend", "search", "user", "reservation", "consul", "jaeger",
        "memcached-reserve", "mongodb-user", "mongodb-reservation")
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


def tool_definitions():
    properties = {
        "inspect": {"resource": {"type": "string", "enum": TOOL_SPEC["inspect"]["resources"]},
                    "name": {"type": "string", "enum": list(APPS)}},
        "logs": {"deployment": {"type": "string", "enum": list(APPS)},
                 "lines": {"type": "integer", "minimum": 1, "maximum": 40}},
        "set_selector": {"service": {"type": "string", "enum": list(APPS)},
                         "selector": {"type": "object", "minProperties": 1, "maxProperties": 8,
                                      "additionalProperties": {"type": "string"}}},
        "restart": {"deployment": {"type": "string", "enum": list(APPS)}},
        "check": {}, "wait": {"seconds": {"type": "integer", "minimum": 1, "maximum": 3}},
        "finish": {"summary": {"type": "string", "maxLength": 256}},
    }
    descriptions = {"inspect": "Inspect one application resource.", "logs": "Read bounded application logs.",
                    "set_selector": "Replace a Service selector.", "restart": "Request a deployment restart.",
                    "check": "Try a fresh booking and report its HTTP response.", "wait": "Wait for an operation to take effect.",
                    "finish": "Declare the investigation complete."}
    return [{"name": name, "description": descriptions[name], "parameters": {
        "type": "object", "properties": properties[name], "required": spec["required"], "additionalProperties": False}}
        for name, spec in TOOL_SPEC.items()]


def validate_call(call):
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        raise ToolInputError("expected tool and arguments only")
    name, args = call["tool"], call["arguments"]
    if not isinstance(name, str) or name not in TOOL_SPEC or not isinstance(args, dict):
        raise ToolInputError("unknown tool or invalid arguments")
    if set(args) != set(TOOL_SPEC[name]["required"]):
        raise ToolInputError("arguments do not match the tool contract")
    for key in ("name", "service", "deployment"):
        if key in args and (not isinstance(args[key], str) or args[key] not in APPS):
            raise ToolInputError("resource is not available")
    if name == "inspect" and args["resource"] not in TOOL_SPEC[name]["resources"]:
        raise ToolInputError("resource type is not available")
    if name in ("logs", "wait"):
        key, low, high = ("lines", 1, 40) if name == "logs" else ("seconds", 1, 3)
        if type(args[key]) is not int or not low <= args[key] <= high:
            raise ToolInputError("argument is outside its allowed range")
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
    def __init__(self, backend, check, record, started, max_calls=16, complete=None):
        self.backend, self.check, self.record = backend, check, record
        self.started, self.max_calls = started, max_calls
        self.calls = 0
        self.checks = 0
        self.mutations = 0
        self.finished_at = None
        self.complete = complete

    def invoke(self, call):
        self.calls += 1
        event = {"call": call, "started_s": time.monotonic() - self.started}
        try:
            if self.calls > self.max_calls:
                raise ToolInputError("tool call budget exhausted")
            if self.finished_at is not None:
                raise ToolInputError("episode already declared complete")
            name, args = validate_call(call)
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
            else:
                response = self.backend.perform(name, args)
                if name in ("set_selector", "restart"):
                    self.mutations += 1
            result = {"ok": True, "result": response}
        except ToolInputError as exc:
            result = {"ok": False, "error": {"code": "invalid_request", "message": str(exc)}}
        except Exception:
            result = {"ok": False, "error": {"code": "operation_failed", "message": "operation failed; inspect application state"}}
        event.update({"response": result, "finished_s": time.monotonic() - self.started})
        self.record(event)
        return result


def operator_role(namespace):
    return [
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
