import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest

from service_operator import retry


CATALOG = {"7": {"name": "hotel seven", "address": {"lat": 38.02, "lon": -122.09}}}
RESPONSE = {"status": 200, "elapsed_s": 0.1, "deadline_s": 4,
            "body": {"type": "FeatureCollection", "features": [
                {"type": "Feature", "id": "7", "geometry": {"type": "Point", "coordinates": [-122.09, 38.02]},
                 "properties": {"name": "hotel seven"}}]}}


def test_search_selects_a_real_returned_hotel():
    assert retry.select_hotel(RESPONSE, CATALOG) == "7"


@pytest.mark.parametrize("damage", ["empty", "unknown", "duplicate", "geometry", "location", "name", "late", "nan"])
def test_search_rejects_false_success(damage):
    response = copy.deepcopy(RESPONSE)
    feature = response["body"]["features"][0]
    if damage == "empty":
        response["body"]["features"] = []
    elif damage == "unknown":
        feature["id"] = "99"
    elif damage == "duplicate":
        response["body"]["features"].append(copy.deepcopy(feature))
    elif damage == "geometry":
        feature["geometry"] = None
    elif damage == "location":
        feature["geometry"]["coordinates"].reverse()
    elif damage == "name":
        feature["properties"] = None
    elif damage == "late":
        response["elapsed_s"] = 5
    else:
        response["elapsed_s"] = float("nan")
    with pytest.raises(ValueError):
        retry.select_hotel(response, CATALOG)


@pytest.mark.parametrize("before,after", [({}, {"n": 1}), ({"n": 2}, {"n": 1}),
                                        ({"n": 0}, {"n": float("nan")}), ({"n": 0}, {"n": "1"})])
def test_missing_reset_or_invalid_counters_are_not_recovery(before, after):
    with pytest.raises(ValueError):
        retry.delta(before, after, "n")


def test_window_counts_actual_starts_and_keeps_missing_outcomes():
    before = {"search_requests_total": 10, "search_rate_attempts_total": 10,
              "rate_requests_expired_total": 0, "rate_requests_completed_total": 10, "rate_queue_depth": 0}
    after = {**before, "search_requests_total": 12, "search_rate_attempts_total": 12,
             "rate_requests_completed_total": 12}
    events = [{"kind": "search", "started_s": 10, "finished_s": 10.1, "response": RESPONSE},
              {"kind": "search", "started_s": 11},
              {"kind": "search", "started_s": 12, "finished_s": 12.1, "response": RESPONSE}]
    result = retry.window(events, 10, 12, CATALOG, before, after)
    assert result["actual_started"] == 2
    assert result["completed"] == result["valid"] == 1
    assert result["success_rate"] == 0.5
    assert not retry.healthy(result)


def test_backlog_alone_does_not_prove_retry_collapse():
    sample = {"actual_rate": 8, "actual_started": 160, "completed": 160, "queue_before": 100,
              "queue_after": 100, "attempts_per_request": 3, "success_rate": 0, "expired_work": 100}
    assert retry.collapsed(sample)
    assert not retry.collapsed({**sample, "actual_rate": 3})
    assert not retry.collapsed({**sample, "attempts_per_request": 1})
    assert not retry.collapsed({**sample, "queue_after": 50})
    assert not retry.collapsed({**sample, "expired_work": 0})


@pytest.mark.parametrize("qps,queue,expected", [(50, 256, True), (0, 256, False), (501, 256, False),
                                              (50, 257, False), (50, float("nan"), False)])
def test_repair_cannot_remove_the_queue_or_capacity_limits(qps, queue, expected):
    assert retry.safe_policy({"rate_backend_qps_limit": qps, "rate_queue_capacity": queue}) is expected


def test_expanded_profile_initializes_real_databases_without_probe_pods():
    aliases = {name: name + ":pinned" for name in ("app", "mongo", "consul", "jaeger", "memcached")}
    objects = retry.profile("owned", aliases, {"geo": "geo init", "rate": "rate init"})
    deployments = {obj["metadata"]["name"]: obj for obj in objects if obj["kind"] == "Deployment"}
    assert len(deployments) == 17
    assert "client" not in deployments and "false-response" not in deployments
    for name in ("geo", "rate"):
        pod = deployments["mongodb-" + name]["spec"]["template"]["spec"]
        command = pod["containers"][0]["command"]
        assert command[:2] == ["/usr/local/bin/docker-entrypoint.sh", "mongod"]
        assert "--auth" in command
        assert any(volume.get("configMap", {}).get("name") == "mongo-init-" + name for volume in pod["volumes"])
    for name in ("search", "rate"):
        service = next(obj for obj in objects if obj["kind"] == "Service" and obj["metadata"]["name"] == name)
        assert len(service["spec"]["ports"]) == 2
        assert all(port.get("name") for port in service["spec"]["ports"])


def test_workload_source_is_checked_before_it_can_execute(tmp_path):
    source = tmp_path / "untrusted.py"
    source.write_text('raise AssertionError("must not execute")\n')
    with pytest.raises(ValueError, match="pinned"):
        retry.load_workload(source, None)


def test_workload_recorder_rejects_external_requests(tmp_path):
    recorder = retry.RecordedRequests(tmp_path / "http.jsonl")
    with pytest.raises(ValueError, match="escaped"):
        recorder.get("https://example.com/hotels", timeout=4)
    assert not recorder.events


def test_unresponsive_stats_process_is_killed_after_timeout():
    path = Path(__file__).parents[1] / "experiments/service_retry_audit.py"
    spec = importlib.util.spec_from_file_location("retry_audit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Stats:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("stats", timeout)

        def kill(self):
            self.killed = True

    stats = Stats()
    module.stop_stats(stats)
    assert stats.terminated and stats.killed
