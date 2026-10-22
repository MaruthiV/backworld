import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from service_operator import retry_episode, tools


RATE = {"qps_limit": 50, "queue_capacity": 256}
SEARCH = {"timeout_ms": 1000, "max_attempts": 2, "backoff_ms": 50, "backoff_multiplier": 2.0, "jitter": 0.2}


class Backend:
    def __init__(self):
        self.calls = []

    def perform(self, name, args):
        self.calls.append((name, args))
        return {"requested": True}


def session(monkeypatch=None):
    backend = Backend()
    events = []
    public = tools.PublicTools(backend, lambda: {"search": {"status": 500}, "booking": None}, events.append,
                               time.monotonic(), profile="retry", deadline_s=300, max_calls=64)
    return public, backend, events


@pytest.mark.parametrize("name,args", [
    ("configure_rate", {**RATE, "qps_limit": True}),
    ("configure_rate", {**RATE, "qps_limit": 501}),
    ("configure_rate", {**RATE, "queue_capacity": 0}),
    ("configure_rate", {**RATE, "queue_capacity": 257}),
    ("configure_rate", {**RATE, "env": {"GOMAXPROCS": "0"}}),
    ("configure_search", {**SEARCH, "max_attempts": 1.5}),
    ("configure_search", {**SEARCH, "backoff_ms": 0}),
    ("configure_search", {**SEARCH, "jitter": float("nan")}),
    ("configure_search", {**SEARCH, "backoff_multiplier": float("inf")}),
    ("metrics", {"service": "mongodb-reservation"}),
    ("metrics", {"service": "search", "namespace": "observer"}),
    ("restart", {"deployment": "mongodb-reservation"}),
    ("set_selector", {"service": "search", "selector": {"app": "other"}}),
])
def test_retry_boundary_rejects_unavailable_fields_and_actions(name, args):
    public, backend, events = session()
    result = public.invoke({"tool": name, "arguments": args})
    assert not result["ok"] and not backend.calls
    assert 0 <= result["clock"]["remaining_s"] <= 300
    assert len(events) == 1


def test_policy_schema_has_no_secret_repair_preset():
    definitions = {row["name"]: row for row in tools.tool_definitions("retry")}
    assert "set_selector" not in definitions
    assert definitions["configure_rate"]["parameters"]["properties"]["qps_limit"]["maximum"] == 500
    assert definitions["metrics"]["parameters"]["properties"]["service"]["enum"] == ["search", "rate"]
    assert "configure_rate" not in {row["name"] for row in tools.tool_definitions()}


def test_expired_actions_cannot_mutate_or_finish(monkeypatch):
    public, backend, _ = session()
    monkeypatch.setattr(tools.time, "monotonic", lambda: public.started + 300)
    for name, args in [("configure_rate", RATE), ("finish", {"summary": "done"})]:
        result = public.invoke({"tool": name, "arguments": args})
        assert not result["ok"] and result["clock"]["remaining_s"] == 0
    assert not backend.calls and public.finished_at is None


def test_clock_charges_backend_latency_and_closes_after_finish(monkeypatch):
    public, backend, _ = session()
    clock = [public.started]
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])

    def perform(name, args):
        clock[0] += 12
        return {"requested": True}

    backend.perform = perform
    result = public.invoke({"tool": "configure_rate", "arguments": RATE})
    assert result["clock"]["elapsed_s"] == 12 and result["clock"]["remaining_s"] == 288
    assert public.mutation_attempts == public.mutations == 1
    result = public.invoke({"tool": "finish", "arguments": {"summary": "done"}})
    assert result["clock"]["closed"] and "success" not in result["result"]
    assert not public.invoke({"tool": "metrics", "arguments": {"service": "rate"}})["ok"]


def test_uncertain_mutation_still_counts_as_an_attempt():
    public, backend, _ = session()

    def fail(name, args):
        raise RuntimeError("private answer /judge/expected.json")

    backend.perform = fail
    result = public.invoke({"tool": "configure_rate", "arguments": RATE})
    assert public.mutation_attempts == 1 and public.mutations == 0
    assert "judge" not in json.dumps(result)


@pytest.mark.parametrize("condition", ["healthy", "repair", "restart_only", "premature_finish"])
def test_control_scripts_reach_completion_through_the_public_contract(condition, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import service_retry_episode_audit

    public, backend, events = session()

    def perform(name, args):
        if name == "inspect":
            return {"generation": 1, "status": {"replicas": 1, "updatedReplicas": 1,
                    "readyReplicas": 1, "observedGeneration": 1}}
        return {"requested": True}

    backend.perform = perform
    monkeypatch.setattr(tools.time, "sleep", lambda seconds: None)
    service_retry_episode_audit.public_script(public, condition)
    assert public.finished_at is not None and public.calls <= 64 and public.checks <= 2
    assert events[-1]["response"]["clock"]["closed"] and all(event["response"]["ok"] for event in events)


def test_policy_patch_contains_only_the_declared_env_fields():
    calls = []
    runtime = SimpleNamespace(kube=lambda args, **kwargs: calls.append(args) or "patched")
    backend = retry_episode.RetryBackend(runtime, "app")
    backend.perform("configure_search", SEARCH)
    command = calls[0]
    assert "--as=system:serviceaccount:app:operator" in command and "--type=strategic" in command
    patch = json.loads(command[-1])
    container = patch["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "search"
    assert {v["name"] for v in container["env"]} == {v[0] for v in tools.POLICY_FIELDS["configure_search"].values()}
    assert set(container) == {"name", "env"}


def test_metrics_ignore_unlisted_content_and_reject_missing_values():
    text = "\n".join(key + " 1" for key in retry_episode.METRICS["search"])
    raw = [text + "\nprivate_expected_answer 123\n"]
    runtime = SimpleNamespace(kube=lambda args, **kwargs: raw[0])
    backend = retry_episode.RetryBackend(runtime, "app")
    assert set(backend.perform("metrics", {"service": "search"})["values"]) == set(retry_episode.METRICS["search"])
    raw[0] = "search_requests_total 1"
    with pytest.raises(ValueError, match="missing"):
        backend.perform("metrics", {"service": "search"})


def test_grade_requires_replay_and_work_after_completion():
    public, _, _ = session()
    public.invoke({"tool": "finish", "arguments": {"summary": "done"}})
    state = {"invariants_preserved": True, "baseline_valid": True, "final_valid": True}
    assert retry_episode.grade(public, state, True, True, True, True, False, 160)["success"]
    assert not retry_episode.grade(public, state, True, False, True, True, False, 160)["success"]
    assert not retry_episode.grade(public, state, True, True, True, True, False, 0)["success"]
    public.mutation_attempts = 1
    assert not retry_episode.grade(public, state, True, True, True, True, True, 160)["success"]


def test_ledger_detects_prior_type_corruption_even_when_python_equality_matches(tmp_path, monkeypatch):
    before = {"reservations": [{"_id": "old", "number": 1}], "capacity": []}
    after = copy.deepcopy(before)
    after["reservations"][0]["number"] = True
    ledger = retry_episode.SearchLedger.__new__(retry_episode.SearchLedger)
    ledger.before, ledger.output = before, tmp_path
    ledger.snapshot = lambda: after
    ledger.events = [{"expected": None, "search": {"status": 500}}]
    monkeypatch.setattr(retry_episode.time, "sleep", lambda value: None)
    assert not ledger.evaluate({})["invariants_preserved"]
