import base64
import copy
import json
import math
import os
import re
import time
import urllib.parse


RESOURCES = {
    ("v1", "Namespace"): ("namespaces", "/api/v1", False),
    ("v1", "Service"): ("services", "/api/v1", True),
    ("v1", "ServiceAccount"): ("serviceaccounts", "/api/v1", True),
    ("v1", "ConfigMap"): ("configmaps", "/api/v1", True),
    ("apps/v1", "Deployment"): ("deployments.apps", "/apis/apps/v1", True),
    ("apps/v1", "DaemonSet"): ("daemonsets.apps", "/apis/apps/v1", True),
    ("batch/v1", "Job"): ("jobs.batch", "/apis/batch/v1", True),
    ("rbac.authorization.k8s.io/v1", "ClusterRole"): ("clusterroles.rbac.authorization.k8s.io", "/apis/rbac.authorization.k8s.io/v1", False),
    ("rbac.authorization.k8s.io/v1", "ClusterRoleBinding"): ("clusterrolebindings.rbac.authorization.k8s.io", "/apis/rbac.authorization.k8s.io/v1", False),
}


def memory_bytes(value):
    match = re.fullmatch(r"([0-9]+)(Ki|Mi|Gi)?", value)
    if not match:
        raise ValueError("unsupported node memory quantity")
    return int(match[1]) * {None: 1, "Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3}[match[2]]


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def identity(obj):
    resource, prefix, namespaced = RESOURCES[(obj["apiVersion"], obj["kind"])]
    name, namespace = obj["metadata"]["name"], obj["metadata"].get("namespace")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", name):
        raise ValueError("invalid planned resource name")
    if namespaced and not re.fullmatch(r"bw-(astro|check)-[a-f0-9]{10}", namespace or ""):
        raise ValueError("planned object is outside the owned namespaces")
    if not namespaced and namespace:
        raise ValueError("cluster resource unexpectedly has a namespace")
    path = prefix + ("/namespaces/" + namespace if namespaced else "")
    path += "/" + resource.split(".", 1)[0] + "/" + name
    return {"resource": resource, "name": name, "namespace": namespace, "path": path,
            "key": obj["kind"] + ":" + (namespace or "") + ":" + name}


def owned(actual, planned, run_id, uid=None):
    if identity(actual) != identity(planned):
        raise ValueError("resource identity changed")
    metadata = actual["metadata"]
    if metadata.get("labels", {}).get("backworld.run") != run_id or not metadata.get("uid"):
        raise ValueError("resource ownership is missing or changed")
    if uid is not None and metadata["uid"] != uid:
        raise ValueError("resource UID changed")
    return metadata["uid"]


def controller_ready(obj):
    spec, status = obj["spec"], obj.get("status", {})
    if status.get("observedGeneration", 0) < obj["metadata"]["generation"]:
        return False
    if obj["kind"] == "Deployment":
        count = spec.get("replicas", 1)
        return count > 0 and all(status.get(key, 0) == count for key in
                                 ("replicas", "updatedReplicas", "readyReplicas", "availableReplicas"))
    count = status.get("desiredNumberScheduled", 0)
    return count == 1 and status.get("numberReady", 0) == count and status.get("updatedNumberScheduled", 0) == count


def validate_pod(pod, images):
    if pod.get("metadata", {}).get("deletionTimestamp") or pod.get("status", {}).get("phase") not in ("Running", "Succeeded"):
        raise ValueError("pod is not current and running/completed")
    if pod["status"]["phase"] == "Running" and not any(c.get("type") == "Ready" and c.get("status") == "True" for c in pod["status"].get("conditions", [])):
        raise ValueError("running pod is not Ready")
    for group, status_key in (("containers", "containerStatuses"), ("initContainers", "initContainerStatuses")):
        expected = {c["name"]: c["image"] for c in pod["spec"].get(group, [])}
        statuses = pod["status"].get(status_key, [])
        if len(statuses) != len(expected) or {s["name"] for s in statuses} != set(expected):
            raise ValueError("container status inventory differs")
        for status in statuses:
            image = expected[status["name"]]
            digest = re.search(r"sha256:[a-f0-9]{64}$", status.get("imageID", ""))
            if image not in images or not digest or digest.group() not in images[image]:
                raise ValueError("runtime image ID differs from the pinned platform")
            if status.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled":
                raise ValueError("pod has an OOM failure")


def validate_application(data, images):
    workloads = data["deployments"] + data["daemonsets"]
    if not workloads or not all(controller_ready(w) for w in workloads):
        raise ValueError("application controllers are not current/ready")
    for pod in data["pods"]:
        validate_pod(pod, images)
    for name in ("frontend-proxy", "frontend", "cart", "product-catalog", "valkey-cart"):
        services = [s for s in data["services"] if s["metadata"]["name"] == name]
        if len(services) != 1:
            raise ValueError("workflow Service is missing or duplicated: " + name)
        validate_routing(services[0], data["endpointslices"], data["pods"], data["replicasets"], data["deployments"])


def validate_routing(service, slices, pods, replicasets, deployments):
    namespace = service["metadata"]["namespace"]
    selector = service["spec"].get("selector", {})
    if not selector or service["spec"].get("type", "ClusterIP") != "ClusterIP":
        raise ValueError("workflow service is not a normal selected ClusterIP")
    by_uid = {p["metadata"]["uid"]: p for p in pods}
    by_rs = {r["metadata"]["uid"]: r for r in replicasets}
    by_deployment = {d["metadata"]["uid"]: d for d in deployments}
    endpoints = []
    for endpoint_slice in slices:
        if endpoint_slice["metadata"].get("labels", {}).get("kubernetes.io/service-name") != service["metadata"]["name"]:
            continue
        owners = endpoint_slice["metadata"].get("ownerReferences", [])
        if not any(o.get("kind") == "Service" and o.get("uid") == service["metadata"]["uid"] for o in owners):
            raise ValueError("EndpointSlice belongs to a different Service")
        for endpoint in endpoint_slice.get("endpoints", []):
            conditions = endpoint.get("conditions", {})
            if conditions.get("ready") is not True or conditions.get("terminating") is True:
                raise ValueError("workflow endpoint is unready or terminating")
            target = endpoint.get("targetRef", {})
            pod = by_uid.get(target.get("uid"))
            if target.get("kind") != "Pod" or target.get("namespace") != namespace or pod is None:
                raise ValueError("workflow endpoint does not name a current app pod")
            if pod["metadata"].get("deletionTimestamp") or any(pod["metadata"]["labels"].get(k) != v for k,v in selector.items()):
                raise ValueError("workflow pod does not match the live selector")
            if not any(c.get("type") == "Ready" and c.get("status") == "True" for c in pod["status"].get("conditions", [])):
                raise ValueError("workflow pod is not Ready")
            rs_uid = next((o["uid"] for o in pod["metadata"].get("ownerReferences", []) if o.get("controller") and o.get("kind") == "ReplicaSet"), None)
            rs = by_rs.get(rs_uid)
            deployment_uid = next((o["uid"] for o in (rs or {}).get("metadata", {}).get("ownerReferences", []) if o.get("controller") and o.get("kind") == "Deployment"), None)
            deployment = by_deployment.get(deployment_uid)
            if not deployment or deployment["metadata"]["name"] != service["metadata"]["name"]:
                raise ValueError("workflow pod belongs to another controller")
            template = copy.deepcopy(rs["spec"]["template"])
            template["metadata"].get("labels", {}).pop("pod-template-hash", None)
            if canonical(template) != canonical(deployment["spec"]["template"]):
                raise ValueError("workflow pod uses a stale controller template")
            if set(endpoint["addresses"]) != {pod["status"]["podIP"]}:
                raise ValueError("endpoint address differs from its Pod")
            endpoints.append(pod["metadata"]["uid"])
    if not endpoints or len(endpoints) != len(set(endpoints)):
        raise ValueError("workflow has missing or duplicate endpoints")


def expected_wire(user, product, quantity):
    user, product = user.encode(), product.encode()
    if not (len(user) < 128 and len(product) < 120 and type(quantity) is int and quantity in (1, 2)):
        raise ValueError("cart exceeds the independent one-item wire check")
    item = b"\x0a" + bytes([len(product)]) + product + b"\x10" + bytes([quantity])
    return (b"\x0a" + bytes([len(user)]) + user + b"\x12" + bytes([len(item)]) + item).hex()


def audit_helper(raw, pod, job, run_id, namespace, hashes):
    status = pod.get("status", {})
    if status.get("phase") != "Succeeded" or job.get("status", {}).get("succeeded") != 1:
        raise ValueError("helper did not complete successfully")
    if not any(o.get("kind") == "Job" and o.get("uid") == job["metadata"]["uid"] and o.get("controller") for o in pod["metadata"].get("ownerReferences", [])):
        raise ValueError("helper Pod belongs to another Job")
    for key in ("containerStatuses", "initContainerStatuses"):
        values = status.get(key, [])
        if len(values) != 1 or values[0].get("restartCount") != 0 or values[0].get("state", {}).get("terminated", {}).get("exitCode") != 0:
            raise ValueError("helper/init exit or retry status is invalid")
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    manifests = [e["manifest"] for e in events if e.get("stage") == "manifest"]
    summaries = [e["summary"] for e in events if e.get("stage") == "summary"]
    if len(manifests) != 1 or len(summaries) != 1:
        raise ValueError("helper logs lack one manifest and summary")
    manifest, summary = manifests[0], summaries[0]
    if manifest["run_id"] != run_id or manifest["namespace"] != namespace or manifest["source_sha256"] != hashes or manifest.get("readiness_only") is not False:
        raise ValueError("helper manifest differs from the shipped control")
    if summary.get("passed") is not True or summary.get("http_requests") != 25 or summary.get("continued_reads") != 20:
        raise ValueError("helper summary did not meet the frozen control")
    users = ["bw-cart-" + run_id + suffix for suffix in ("-prior", "-current")]
    intents, completed, written, stored, observed = {}, set(), {}, {}, {}
    absent, stored_positions, last_response_position = set(), {}, -1
    catalog = None
    first_product = None
    for position, event in enumerate(events):
        stage = event.get("stage")
        if stage == "intent":
            request_id = event["request_id"]
            if request_id != run_id + f"-{len(intents) + 1:03d}":
                raise ValueError("request intent IDs are missing or duplicated")
            intents[request_id] = event
            if event["method"] == "POST":
                if absent != set(users):
                    raise ValueError("both fresh owned carts were not independently observed")
                body = event["body"]
                index = len(written)
                if index >= 2 or canonical(body) != canonical({"userId": users[index], "item": {"productId": first_product, "quantity": 2 - index}}):
                    raise ValueError("unexpected or repeated cart mutation")
                written[users[index]] = {"userId": users[index], "items": [body["item"]]}
        elif stage == "response":
            last_response_position = position
            request_id, response = event["request_id"], event["response"]
            if request_id not in intents or request_id in completed:
                raise ValueError("response has no unique prior intent")
            completed.add(request_id)
            elapsed, deadline = response.get("elapsed_s"), response.get("deadline_s")
            if (response.get("status") != 200 or response.get("error") or response.get("request_id") != request_id
                    or type(elapsed) not in (int, float) or type(deadline) not in (int, float)
                    or not math.isfinite(elapsed) or not 0 <= elapsed <= deadline <= 5):
                raise ValueError("helper response was not timely and valid")
            decoded = json.loads(base64.b64decode(response["raw_body_b64"], validate=True))
            if canonical(decoded) != canonical(response["body"]):
                raise ValueError("raw and parsed response bodies disagree")
            intent, body = intents[request_id], response["body"]
            if intent["method"] == "POST" and intent["path"] == "/api/cart":
                expected = written[intent["body"]["userId"]]
            elif intent["method"] == "GET" and intent["path"] == "/api/products?currencyCode=USD":
                if not isinstance(body, list) or not body or any(not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p.get("name") or not p.get("priceUsd") for p in body):
                    raise ValueError("catalog is not useful")
                indexed = {p["id"]: p for p in body}
                if len(indexed) != len(body):
                    raise ValueError("catalog contains duplicate products")
                if catalog is None:
                    catalog, first_product = indexed, body[0]["id"]
                if canonical(indexed) != canonical(catalog):
                    raise ValueError("catalog changed during a healthy control")
                expected = body
            elif intent["method"] == "GET" and urllib.parse.urlsplit(intent["path"]).path == "/api/cart":
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(intent["path"]).query)
                if set(query) != {"sessionId", "currencyCode"} or query["currencyCode"] != ["USD"] or len(query["sessionId"]) != 1:
                    raise ValueError("unexpected cart read")
                expected = copy.deepcopy(written[query["sessionId"][0]])
                expected["items"][0]["product"] = catalog[first_product]
            else:
                raise ValueError("unexpected workflow request")
            if canonical(body) != canonical(expected):
                raise ValueError("response disagrees with the independent workflow check")
        elif stage == "store":
            value, stamp = event["snapshot"], event["observed_monotonic_s"]
            if type(stamp) not in (int, float) or not math.isfinite(stamp):
                raise ValueError("store observation has an invalid clock")
            user = value["user"]
            if user not in users:
                raise ValueError("unexpected private cart key")
            if user not in written:
                if value != {"user": user, "key_type": "none", "hash_fields": 0, "ttl_ms": -2, "raw_hex": None, "cart": None}:
                    raise ValueError("control did not start from absent owned carts")
                absent.add(user)
            else:
                expected = written[user]
                ttl = value["ttl_ms"]
                if (value["key_type"] != "hash" or value["hash_fields"] != 1 or type(ttl) is not int or not 0 < ttl <= 3600000
                        or value["raw_hex"] != expected_wire(user, first_product, expected["items"][0]["quantity"])
                        or canonical(value["cart"]) != canonical(expected)):
                    raise ValueError("stored bytes or expiry differ from the independent cart check")
                if user in stored:
                    delta = (stamp - observed[user]) * 1000
                    if ttl > stored[user]["ttl_ms"] or abs(stored[user]["ttl_ms"] - ttl - delta) > 5000:
                        raise ValueError("reads changed expiry or clocks disagree")
                stored[user], observed[user] = value, stamp
                stored_positions[user] = position
    if len(intents) != 25 or completed != set(intents) or set(stored) != set(users) or len(written) != 2:
        raise ValueError("helper work or stored-state evidence is incomplete")
    if any(stored_positions[user] <= last_response_position for user in users):
        raise ValueError("final cart bytes were not observed after continued work")
    return {"passed": True, "requests": 25, "write_attempts": 2, "independent_wire_checks": True}


class KubeControl:
    def __init__(self, command, prefix, objects, output, run_id):
        if "--kubeconfig" not in prefix or "--context" not in prefix:
            raise ValueError("an explicit kubeconfig and context are required")
        self.command, self.prefix, self.objects = command, prefix, objects
        self.output, self.run_id = output, run_id
        self.receipts, self.intents = {}, {}
        self.allowed = {identity(obj)["key"]: obj for obj in objects}
        if len(self.allowed) != len(objects):
            raise ValueError("duplicate planned resource identities")

    def get(self, planned):
        target = identity(planned)
        if target["key"] not in self.allowed:
            raise ValueError("resource is outside the fixed package")
        args = ["get", target["resource"], target["name"], "--ignore-not-found", "-o", "json"]
        if target["namespace"]:
            args += ["--namespace", target["namespace"]]
        raw = self.command(self.prefix + args)
        return json.loads(raw) if raw.strip() else None

    def save(self):
        state = {"run_id": self.run_id, "intents": self.intents, "receipts": self.receipts}
        temporary = self.output / "ownership.tmp"
        with temporary.open("w") as handle:
            handle.write(json.dumps(state, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.output / "ownership.json")

    def create(self, objects):
        for obj in objects:
            if canonical(obj) != canonical(self.allowed[identity(obj)["key"]]):
                raise ValueError("creation differs from the fixed package")
            owned({**obj, "metadata": {**obj["metadata"], "uid": "planned"}}, obj, self.run_id)
            if self.get(obj) is not None:
                raise ValueError("planned resource already exists; refusing adoption")
        for obj in objects:
            key = identity(obj)["key"]
            self.intents[key] = {"planned": obj, "started_unix_s": time.time()}
            self.save()
            try:
                raw = self.command(self.prefix + ["create", "-f", "-", "-o", "json"], input=json.dumps(obj))
                actual = json.loads(raw)
                self.receipts[key] = {"uid": owned(actual, obj, self.run_id), "planned": obj}
                self.save()
            except Exception:
                actual = self.get(obj)
                if actual is not None:
                    self.receipts[key] = {"uid": owned(actual, obj, self.run_id), "planned": obj, "uncertain_create": True}
                    self.save()
                raise

    def cleanup(self, roots):
        errors = []
        for obj in roots:
            target = identity(obj)
            try:
                actual = self.get(obj)
                receipt = self.receipts.get(target["key"])
                if actual is None:
                    continue
                if receipt is None:
                    if target["key"] not in self.intents:
                        raise ValueError("cleanup has no prior creation intent")
                    receipt = {"uid": owned(actual, obj, self.run_id), "planned": obj, "uncertain_create": True}
                    self.receipts[target["key"]] = receipt
                    self.save()
                owned(actual, obj, self.run_id, receipt["uid"])
                body = {"apiVersion": "v1", "kind": "DeleteOptions", "preconditions": {"uid": receipt["uid"]}, "propagationPolicy": "Foreground"}
                self.command(self.prefix + ["delete", "--raw", target["path"], "-f", "-"], input=json.dumps(body))
            except Exception as exc:
                errors.append(target["key"] + ": " + str(exc))
        deadline = time.monotonic() + 90
        remaining = []
        while True:
            try:
                remaining = [identity(obj)["key"] for obj in roots if self.get(obj) is not None]
            except Exception as exc:
                errors.append("absence check failed: " + str(exc))
                remaining = [identity(obj)["key"] for obj in roots]
                break
            if not remaining or time.monotonic() >= deadline:
                break
            time.sleep(1)
        return {"ok": not errors and not remaining, "errors": errors, "remaining": remaining}
