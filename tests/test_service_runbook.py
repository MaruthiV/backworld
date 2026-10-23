import inspect
import time
from types import SimpleNamespace

import pytest

from service_operator import pilot, runbook


def private_grade(case="selector", **overrides):
    public = SimpleNamespace(started=time.monotonic(), finished_at=None, deadline_s=300,
                             mutation_attempts=0, mutations=0, calls=1, checks=0)
    public.finished_at = public.started + 1
    args = {"case": case, "public": public,
            "evaluation": {"invariants_preserved": True, "baseline_valid": True, "final_valid": True},
            "recovered": True, "replayed": None, "topology_ok": True,
            "safe_policy": True, "post_completion": 160}
    args.update(overrides)
    return pilot.grade(**args)


def test_selector_does_not_need_a_fabricated_replay():
    assert private_grade()["success"]
    assert not private_grade()["replay_required"]
    with pytest.raises(ValueError, match="only defined"):
        private_grade(replayed=True)
    assert not private_grade("retry")["success"]
    assert private_grade("retry", replayed=True)["success"]


@pytest.mark.parametrize("field", ["invariants_preserved", "baseline_valid", "final_valid"])
def test_successful_traffic_cannot_hide_failed_database_checks(field):
    state = {"invariants_preserved": True, "baseline_valid": True, "final_valid": True, field: False}
    assert not private_grade(evaluation=state)["success"]


def test_healthy_must_not_try_mutations_even_if_the_operation_failed():
    public = SimpleNamespace(started=10, finished_at=11, deadline_s=300,
                             mutation_attempts=1, mutations=0, calls=2, checks=0)
    assert not private_grade("healthy", public=public)["success"]
    public.mutation_attempts = 0
    assert private_grade("healthy", public=public)["success"]
    public.finished_at = 310
    assert not private_grade("healthy", public=public)["success"]


@pytest.mark.parametrize("changes", [{"recovered": False}, {"topology_ok": False},
                                      {"safe_policy": False}, {"post_completion": 0}])
def test_finish_alone_never_proves_recovery(changes):
    assert not private_grade(**changes)["success"]


class Observations:
    def __init__(self, bad_route=False, queue=0, reset_counters=False):
        self.bad_route, self.queue, self.reset_counters = bad_route, queue, reset_counters
        self.calls, self.count = [], 0

    def send(self, call):
        name, args = call["tool"], call["arguments"]
        self.calls.append(call)
        result = {}
        if name == "inspect":
            resource = args["resource"]
            if resource == "service":
                result = {"selector": {"route": "shared"} if self.bad_route else {"app": "frontend"}}
            elif resource == "deployment":
                result = {"generation": 2, "pod_labels": {"app": args["name"], "route": "shared"},
                          "status": {"replicas": 1, "updatedReplicas": 1, "readyReplicas": 1, "observedGeneration": 2}}
            elif resource == "pods":
                result = {"pods": [{"name": "front-pod", "conditions": [{"type": "Ready", "status": "True"}]}]}
            elif resource == "endpoints":
                names = ["front-pod", "search-pod"] if self.bad_route else ["front-pod"]
                result = {"slices": [{"endpoints": [{"targetRef": {"name": value}, "conditions": {"ready": True}}
                                                     for value in names]}]}
        elif name == "set_selector":
            assert args["selector"] == {"app": "frontend"}
            self.bad_route = False
        elif name == "metrics":
            if args["service"] == "search":
                self.count += -40 if self.reset_counters else 40
                result = {"values": {"search_requests_total": self.count,
                                     "search_rate_attempts_total": self.count * (3 if self.queue else 1)}}
            else:
                result = {"values": {"rate_queue_depth": self.queue}}
        elif name == "configure_search":
            self.queue = 0
            self.count = 0
        elif name == "check":
            result = {"search": {"status": 200, "body": {"features": [{"id": "4"}]}},
                      "booking": {"status": 200}}
        return {"ok": True, "result": result}


def test_runbook_takes_only_a_public_sender_and_leaves_healthy_service_alone():
    assert list(inspect.signature(runbook.investigate).parameters) == ["send"]
    observations = Observations()
    runbook.investigate(observations.send)
    names = [row["tool"] for row in observations.calls]
    assert names[-1] == "finish" and names.count("check") == 1
    assert not {"set_selector", "restart", "configure_rate", "configure_search"}.intersection(names)


def test_runbook_repairs_only_the_observed_routing_mismatch():
    observations = Observations(bad_route=True)
    runbook.investigate(observations.send)
    names = [row["tool"] for row in observations.calls]
    assert names.count("set_selector") == 1 and "configure_rate" not in names
    assert names[-1] == "finish"


def test_runbook_uses_counter_deltas_and_queue_to_apply_policy():
    observations = Observations(queue=240)
    runbook.investigate(observations.send)
    names = [row["tool"] for row in observations.calls]
    assert "configure_rate" in names and "configure_search" in names and "set_selector" not in names
    assert names[-1] == "finish"


@pytest.mark.parametrize("observations", [Observations(queue=10), Observations(reset_counters=True)])
def test_runbook_leaves_unresolved_observations_unfinished(observations):
    with pytest.raises(RuntimeError):
        runbook.investigate(observations.send)
    assert not any(row["tool"] in ("finish", "configure_rate", "set_selector") for row in observations.calls)
