from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backworld import OrderWorld
from experiments.importance_noise import sample_fixed_counts


def test_sample_counts_and_concentration_stats_do_not_depend_on_elapsed_time(monkeypatch):
    class FakeSampler:
        def __init__(self, *args):
            self.policy_evaluations = self.transitions = self.sampled_steps = 0
            self.proposals = {}
            self.bounds = {}
            self.last_log_weight = 0.0

        def estimate(self, rng):
            self.sampled_steps += 1
            self.policy_evaluations += 1
            self.transitions += 1
            value = self.sampled_steps / 10
            self.last_log_weight = float(np.log(value))
            return value, np.full((2, 2), value)

    clock = count(0.0, 10.0)
    monkeypatch.setattr("experiments.importance_noise.perf_counter", lambda: next(clock))
    monkeypatch.setattr("backworld.importance.FeasibleImportance", FakeSampler)
    run = sample_fixed_counts(OrderWorld(), np.zeros((2, 2)), 8, [2, 5], 7)
    first, last = run["checkpoints"]
    assert first["samples"] == first["counters"]["sampled_steps"] == 2
    assert last["samples"] == last["counters"]["sampled_steps"] == 5
    assert first["elapsed_seconds"] > 1
    assert last["elapsed_seconds"] > first["elapsed_seconds"]
    assert last["mean_value"] == pytest.approx(0.3)
    assert last["empirical_gradient_variance_trace"] == pytest.approx(0.1)
    assert last["largest_value_share"] == pytest.approx(1 / 3)
    assert last["top_one_percent_value_share"] == pytest.approx(1 / 3)
    assert last["largest_squared_gradient_share"] == pytest.approx(5 / 11)
    assert last["nonfinite_log_weights"] == 0


@pytest.mark.parametrize("moment_limit", [1, 5_000])
def test_fixed_count_worker_keeps_moment_limits_separate_from_sampling(tmp_path, moment_limit):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "noise.json"
    completed = subprocess.run([
        sys.executable, "-m", "experiments.importance_noise",
        "--orders", "1", "--variants", "shared", "--seeds", "7",
        "--counts", "2", "5", "--repetitions", "1",
        "--moment-max-nodes", str(moment_limit), "--output", str(output),
    ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    case, = report["cases"]
    assert case["status"] == "complete"
    assert case["moments"]["status"] == ("complete" if moment_limit > 1 else "resource_limit")
    run, = case["runs"]
    assert [point["samples"] for point in run["checkpoints"]] == [2, 5]
    for point in run["checkpoints"]:
        assert point["counters"]["sampled_steps"] == 4 * point["samples"]
        expected_error = point["expected_relative_squared_gradient_error"]
        assert (expected_error is not None) == (moment_limit > 1)
