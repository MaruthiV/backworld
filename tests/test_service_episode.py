import time
import subprocess
from types import SimpleNamespace

from service_operator import http_probe
from service_operator.episode import PrivateWorkload, grade_episode
from service_operator.traffic import Traffic


def test_completion_does_not_cancel_future_work(tmp_path, monkeypatch):
    monkeypatch.setattr(http_probe, "request", lambda url, deadline: {"status": 200})
    driver = Traffic("http://unused", tmp_path / "events.jsonl", count=4, interval=0.03)
    driver.finish()
    driver.start()
    deadline = time.monotonic() + 2
    while driver.ended is None and time.monotonic() < deadline:
        time.sleep(0.01)
    state = driver.status()
    assert state["ended_monotonic_s"] is not None
    assert len(state["events"]) == 4
    assert len(state["journal"]) == 8
    assert all(event["offered_monotonic_s"] > state["declared_complete_monotonic_s"] for event in state["events"])
    for event in state["events"]:
        event["verdict"] = {"accepted": True, "invariant_ok": True}
    worker = SimpleNamespace(events=state["events"], count=4, error=None, driver_state=state)
    tools = SimpleNamespace(finished_at=time.monotonic(), mutations=0)
    ledger = SimpleNamespace(events=state["events"])
    assert grade_episode(worker, ledger, tools, True, True)["success"]
    assert not grade_episode(worker, ledger, tools, False, False)["success"]


def test_missing_workload_samples_cannot_pass():
    worker = SimpleNamespace(events=[], count=12, error="probe failed",
                             driver_state={"declared_complete_monotonic_s": 1})
    tools = SimpleNamespace(finished_at=time.monotonic(), mutations=0)
    result = grade_episode(worker, SimpleNamespace(events=[]), tools, True, True)
    assert not result["success"]
    assert not result["workload_complete"]


def test_diagnostic_and_scheduled_requests_share_one_serial_lane(tmp_path, monkeypatch):
    active = []
    overlap = []

    def request(url, deadline):
        active.append(url)
        overlap.append(len(active))
        time.sleep(0.02)
        active.remove(url)
        return {"status": 200}

    monkeypatch.setattr(http_probe, "request", request)
    driver = Traffic("http://unused", tmp_path / "events.jsonl", count=3, interval=0.03)
    driver.start()
    driver.probe("diagnostic")
    deadline = time.monotonic() + 2
    while driver.ended is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.ended is not None
    assert len(driver.events) == 4 and max(overlap) == 1


def test_collection_retries_only_the_read_after_a_transport_timeout(tmp_path, monkeypatch):
    calls = []

    def rpc(action):
        calls.append(action)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("status", 15)
        return {"ended_monotonic_s": 36, "events": [], "error": None}

    ledger = SimpleNamespace(output=tmp_path, rpc=rpc, evaluate=lambda events, name: events)
    worker = PrivateWorkload(ledger)
    worker.started = time.monotonic() - 40
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    worker.join()
    assert calls == ["status", "status"]
    assert (tmp_path / "collection-errors.json").exists()
