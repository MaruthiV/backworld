import copy

import pytest

from service_operator.selector import check_route


def route():
    state = {key: {"items": []} for key in ("deployments", "pods", "replicasets", "slices")}
    revision = {"deployment.kubernetes.io/revision": "3"}
    for name in ("frontend", "search", "false-response"):
        state["deployments"]["items"].append({
            "metadata": {"name": name, "uid": name + "-dep", "generation": 3, "annotations": revision},
            "spec": {"replicas": 1},
            "status": {"readyReplicas": 1, "updatedReplicas": 1, "observedGeneration": 3},
        })
        state["replicasets"]["items"].append({
            "metadata": {"name": name + "-rs", "uid": name + "-rs-uid", "annotations": copy.copy(revision),
                         "ownerReferences": [{"kind": "Deployment", "name": name, "uid": name + "-dep"}]},
            "spec": {"replicas": 1},
        })
        state["pods"]["items"].append({
            "metadata": {"name": name + "-pod", "labels": {"io.kompose.service": name},
                         "ownerReferences": [{"kind": "ReplicaSet", "name": name + "-rs", "uid": name + "-rs-uid"}]},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        })
    state["slices"]["items"] = [{"ports": [{"port": 5000}], "endpoints": [
        {"conditions": {"ready": True}, "targetRef": {"kind": "Pod", "name": "frontend-pod"}},
    ]}]
    return state


def test_current_ready_frontend_passes():
    assert check_route(route(), ["frontend"])


def test_extra_search_endpoint_is_rejected_as_healthy():
    state = route()
    state["slices"]["items"][0]["endpoints"].append({
        "conditions": {"ready": True}, "targetRef": {"kind": "Pod", "name": "search-pod"},
    })
    assert not check_route(state, ["frontend"])
    assert check_route(state, ["frontend", "search"])


@pytest.mark.parametrize("change", ["missing", "unready", "deleting", "wrong_uid", "old_revision", "scaled_down", "wrong_port"])
def test_invalid_routing_evidence_is_rejected(change):
    state = route()
    if change == "missing":
        state["slices"]["items"][0]["endpoints"] = []
    elif change == "unready":
        state["pods"]["items"][0]["status"]["conditions"][0]["status"] = "False"
    elif change == "deleting":
        state["pods"]["items"][0]["metadata"]["deletionTimestamp"] = "2026-10-21T00:00:00Z"
    elif change == "wrong_uid":
        state["pods"]["items"][0]["metadata"]["ownerReferences"][0]["uid"] = "old-rs-uid"
    elif change == "old_revision":
        state["replicasets"]["items"][0]["metadata"]["annotations"]["deployment.kubernetes.io/revision"] = "2"
    elif change == "scaled_down":
        state["deployments"]["items"][0]["spec"]["replicas"] = 0
    else:
        state["slices"]["items"][0]["ports"][0]["port"] = 8082
    assert not check_route(state, ["frontend"])
