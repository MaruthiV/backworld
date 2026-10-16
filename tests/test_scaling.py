import numpy as np
import pytest

from backworld import OrderWorld, SoftmaxPolicy, compile_world
from backworld.environment import OrderState
from backworld.estimators import ExactSuffix
from backworld.graph import GraphLimitError
from backworld.oracle import exact_forward
from backworld.planning import shortest_plan
from backworld.policy import initial_weights, observation


@pytest.mark.parametrize("count", [1, 3, 4])
def test_every_order_has_distinct_actions_and_stock(count):
    world = OrderWorld(shared_inventory=False, replenishments=0, target_orders=(True,) * count)
    state = world.initial_state()
    assert state.stock == (1,) * count
    for order in range(count):
        for operation in range(3):
            state = world.step(state, 4 * order + operation)
        assert state.stages[order] == 3
        assert bool(world.reward(state)) == (order == count - 1)
    assert state.stock == (0,) * count
    assert world.step(state, world.action_count - 1) == state


def test_three_orders_compete_for_shared_stock():
    world = OrderWorld(target_orders=(True, True, True), replenishments=2)
    state = world.step(world.initial_state(), 8)
    assert state == OrderState((0, 0, 1), (0, 0, 0), 2)
    assert world.step(state, 0) == state
    state = world.step(state, world.action_count - 2)
    assert state.stock == (1, 0, 0)
    state = world.step(state, 11)
    assert state == OrderState((0, 0, 0), (2, 0, 0), 1)


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("shared", [True, False])
def test_multiorder_backward_matches_the_independent_reference(count, shared):
    world = OrderWorld(
        target_orders=(True,) * count, shared_inventory=shared,
        replenishments=count - 1 if shared else 0,
    )
    horizon = 4 * count
    weights = initial_weights(19, world)
    assert weights.shape == (4 * count + 2, 6 * count + 4)
    assert observation(world, world.initial_state(), horizon, horizon).shape == (6 * count + 4,)
    reference = exact_forward(world, weights, horizon)
    policy = SoftmaxPolicy(weights)
    value = compile_world(world, horizon).expected_return(policy)
    value.backward()
    assert reference.value > 0
    assert np.linalg.norm(reference.gradient) > 0
    assert float(value.detach()) == pytest.approx(reference.value, rel=1e-12)
    np.testing.assert_allclose(policy.weights.grad.numpy(), reference.gradient, atol=1e-15, rtol=1e-10)


def test_full_suffix_uses_all_multiorder_actions():
    world = OrderWorld(shared_inventory=False, replenishments=0, target_orders=(True,) * 3)
    weights = initial_weights(7, world)
    reference = exact_forward(world, weights, 9)
    value, gradient = ExactSuffix(world, weights, 9, 9).estimate(np.random.default_rng(0))
    assert value == pytest.approx(reference.value, rel=1e-12)
    np.testing.assert_allclose(gradient, reference.gradient, atol=1e-15, rtol=1e-10)


def test_weights_from_a_different_order_count_are_rejected():
    world = OrderWorld(target_orders=(True,) * 3)
    weights = initial_weights()
    with pytest.raises(ValueError, match="dimensions"):
        ExactSuffix(world, weights, 12, 2)
    with pytest.raises(ValueError, match="dimensions"):
        exact_forward(world, weights, 12)


def test_empty_order_world_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        OrderWorld(target_orders=())


def test_graph_deadline_reports_a_resource_limit(monkeypatch):
    clock = iter([0.0, 2.0])
    monkeypatch.setattr("backworld.graph.perf_counter", lambda: next(clock))
    with pytest.raises(GraphLimitError) as failure:
        compile_world(OrderWorld(), 8, max_seconds=1.0)
    assert failure.value.limit == "max_seconds"
    assert failure.value.elapsed_seconds == 2.0


@pytest.mark.parametrize("world,length", [
    (OrderWorld(), 7),
    (OrderWorld(shared_inventory=False, replenishments=0, target_orders=(True,) * 3), 9),
    (OrderWorld(target_orders=(True, False)), 3),
])
def test_planner_finds_a_shortest_solution_that_can_wait_until_the_horizon(world, length):
    horizon = length + 2
    result = shortest_plan(world, horizon)
    assert result.status == "solved"
    assert len(result.actions) == length
    state = world.initial_state()
    for action in result.actions + (world.action_count - 1,) * (horizon - length):
        state = world.step(state, action)
    assert world.reward(state) == 1


def test_planner_distinguishes_exhaustion_from_a_node_limit():
    world = OrderWorld()
    exhausted = shortest_plan(world, 2)
    assert exhausted.status == "unreachable_within_horizon"
    limited = shortest_plan(world, 8, max_nodes=1)
    assert limited.status == "node_limit"
    assert limited.actions is None


def test_planner_deadline_does_not_claim_unsolvability(monkeypatch):
    clock = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr("backworld.planning.perf_counter", lambda: next(clock))
    result = shortest_plan(OrderWorld(), 8, max_seconds=1.0)
    assert result.status == "time_limit"
    assert result.actions is None
import json
import os
from pathlib import Path
import subprocess
import sys



def test_scaling_runner_preserves_resource_limits_and_skipped_cases(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "scaling.json"
    completed = subprocess.run([
        sys.executable, "-m", "experiments.scaling_audit",
        "--orders", "2", "3", "--variants", "shared", "--max-nodes", "1",
        "--samples", "1", "--repeats", "1", "--case-seconds", "10",
        "--total-seconds", "15", "--output", str(output),
    ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    first, second = report["cases"]
    assert first["exact"]["status"] == "resource_limit"
    assert first["exact"]["success_probability"] is None
    assert second["status"] == "skipped"
    assert second["reason"] == "previous_resource_limit"
