import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backworld import OrderWorld
from experiments.importance_audit import timed_estimates


def test_budget_checkpoints_exclude_calls_that_finish_late(monkeypatch):
    class FakeSampler:
        def __init__(self, *args):
            self.policy_evaluations = self.transitions = self.sampled_steps = 0
            self.proposals = {}
            self.bounds = {}

        def estimate(self, rng):
            self.policy_evaluations += 1
            self.transitions += 1
            self.sampled_steps += 1
            return 1.0, np.ones((2, 2))

    clock = iter([0.0, 0.001, 0.04, 0.06, 0.27, 0.28])
    monkeypatch.setattr("experiments.importance_audit.perf_counter", lambda: next(clock))
    monkeypatch.setattr("backworld.importance.FeasibleImportance", FakeSampler)
    run = timed_estimates(OrderWorld(), np.zeros((2, 2)), 8, "feasible_importance", [0.05, 0.25], 7)
    first, second = run["checkpoints"]
    assert first["samples"] == 1
    assert second["samples"] == 2
    assert first["counters"]["executed_transitions"] == 1
    assert second["counters"]["executed_transitions"] == 2
    assert first["last_included_completion_seconds"] == 0.04
    assert second["last_included_completion_seconds"] == 0.06
    assert first["crossing_call_overshoot_seconds"] == pytest.approx(0.01)
    assert second["crossing_call_overshoot_seconds"] == pytest.approx(0.02)
    assert run["total_samples_including_final_crossing_call"] == 3


@pytest.mark.parametrize("max_nodes", [1, 50_000])
def test_isolated_importance_audit_preserves_costs_and_reference_limits(tmp_path, max_nodes):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "importance.json"
    completed = subprocess.run([
        sys.executable, "-m", "experiments.importance_audit",
        "--orders", "1", "--variants", "shared", "--seeds", "7",
        "--budgets", "0.005", "0.01", "--repetitions", "1",
        "--max-nodes", str(max_nodes), "--output", str(output),
    ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    case, = report["cases"]
    if max_nodes == 1:
        assert case["status"] == "reference_resource_limit"
        assert case["runs"] == []
    else:
        assert case["status"] == "complete"
        assert len(case["runs"]) == 3
        assert case["exact"]["construction_seconds"] > 0
        assert case["exact"]["gradient_norm"] > 0
        for run in case["runs"]:
            assert len(run["checkpoints"]) == 2
            for checkpoint in run["checkpoints"]:
                assert checkpoint["last_included_completion_seconds"] <= checkpoint["budget_seconds"]
                assert checkpoint["crossing_call_completion_seconds"] > checkpoint["budget_seconds"]
