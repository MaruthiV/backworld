import ast
import hashlib
import json


INJECTOR_BLOB = "b38852b4157391857ee0c8f44cdde041c0bc5a3c"


def load_selector_functions(path):
    body = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if blob != INJECTOR_BLOB:
        raise ValueError("upstream injector differs from pinned source")
    tree = ast.parse(body)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "VirtualizationFaultInjector")
    names = ("inject_service_wrong_pod_selection", "recover_service_wrong_pod_selection")
    namespace = {"json": json}
    for name in names:
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def check_route(state, expected):
    deployments = {item["metadata"]["name"]: item for item in state["deployments"]["items"]}
    pods = {item["metadata"]["name"]: item for item in state["pods"]["items"]}
    replica_sets = {item["metadata"]["name"]: item for item in state["replicasets"]["items"]}
    selected = []
    for part in state["slices"]["items"]:
        if any(port.get("port") != 5000 for port in part.get("ports", [])):
            return False
        for endpoint in part.get("endpoints", []):
            if endpoint.get("conditions", {}).get("ready") is not True:
                continue
            ref = endpoint.get("targetRef", {})
            if ref.get("kind") != "Pod" or ref.get("name") not in pods:
                return False
            pod = pods[ref["name"]]
            if pod["metadata"].get("deletionTimestamp"):
                return False
            owners = [owner for owner in pod["metadata"].get("ownerReferences", []) if owner["kind"] == "ReplicaSet"]
            if len(owners) != 1 or owners[0]["name"] not in replica_sets:
                return False
            rs = replica_sets[owners[0]["name"]]
            if owners[0].get("uid") != rs["metadata"].get("uid"):
                return False
            parents = [owner for owner in rs["metadata"].get("ownerReferences", []) if owner["kind"] == "Deployment"]
            if len(parents) != 1:
                return False
            name = parents[0]["name"]
            dep = deployments.get(name, {})
            revision_key = "deployment.kubernetes.io/revision"
            if (parents[0].get("uid") != dep.get("metadata", {}).get("uid")
                    or rs["metadata"].get("annotations", {}).get(revision_key) is None
                    or rs["metadata"].get("annotations", {}).get(revision_key)
                    != dep.get("metadata", {}).get("annotations", {}).get(revision_key)):
                return False
            if (name not in expected or rs.get("spec", {}).get("replicas", 0) < 1
                    or pod["metadata"].get("labels", {}).get("io.kompose.service") != name
                    or not any(c["type"] == "Ready" and c["status"] == "True" for c in pod.get("status", {}).get("conditions", []))):
                return False
            selected.append(name)
    if sorted(selected) != sorted(expected):
        return False
    for name in set(expected) | {"frontend", "search"}:
        dep = deployments.get(name, {})
        status = dep.get("status", {})
        if (dep.get("spec", {}).get("replicas") != 1 or status.get("readyReplicas") != 1
                or status.get("updatedReplicas") != 1
                or status.get("observedGeneration", 0) < dep.get("metadata", {}).get("generation", 1)):
            return False
    return True
