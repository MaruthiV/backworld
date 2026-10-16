from collections import defaultdict, deque

import numpy as np
import pytest

from backworld import OrderWorld, SoftmaxPolicy, compile_world
from backworld.feasibility import completion_lower_bound
from backworld.oracle import exact_forward
from backworld.policy import initial_weights


def exhaustive_distances(world):
    initial = world.initial_state()
    states = {initial}
    pending = deque([initial])
    predecessors = defaultdict(set)
    while pending:
        state = pending.popleft()
        for action in range(world.action_count):
            following = world.step(state, action)
            predecessors[following].add(state)
            if following not in states:
                states.add(following)
                pending.append(following)
    distances = {state: 0 for state in states if world.reward(state)}
    pending = deque(distances)
    while pending:
        state = pending.popleft()
        for previous in predecessors[state]:
            if previous not in distances:
                distances[previous] = distances[state] + 1
                pending.append(previous)
    return states, distances


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("stock,replenishments,targets", [
    (0, 2, (True, True)),
    (0, 2, (False, True)),
    (1, 1, (True, False)),
    (1, 1, (False, False)),
    (2, 0, (True, True)),
    (1, 0, (True, True)),
    (1, 2, (True, False, True)),
])
def test_bound_never_discards_an_exhaustively_reachable_success(shared, stock, replenishments, targets):
    world = OrderWorld(shared, stock, replenishments, targets)
    states, distances = exhaustive_distances(world)
    for state in states:
        bound = completion_lower_bound(world, state)
        if state in distances:
            assert bound is not None
            assert 0 <= bound <= distances[state]
        if bound is None:
            assert state not in distances


@pytest.mark.parametrize("world,horizon", [
    (OrderWorld(), 8),
    (OrderWorld(), 7),
    (OrderWorld(), 6),
    (OrderWorld(), 0),
    (OrderWorld(shared_inventory=False, replenishments=0, target_orders=(True,) * 3), 12),
    (OrderWorld(target_orders=(True, False)), 6),
    (OrderWorld(target_orders=(False, False)), 4),
    (OrderWorld(target_orders=(False, False)), 0),
    (OrderWorld(initial_stock=0, replenishments=2), 8),
    (OrderWorld(False, 0, 3, (False, True)), 8),
    (OrderWorld(initial_stock=2, replenishments=0), 6),
])
def test_pruned_values_and_gradients_match_the_unpruned_oracle(world, horizon):
    weights = initial_weights(31, world)
    reference = exact_forward(world, weights, horizon)
    for prune in (False, True):
        graph = compile_world(world, horizon, prune_impossible=prune)
        policy = SoftmaxPolicy(weights)
        value = graph.expected_return(policy)
        value.backward()
        assert float(value.detach()) == pytest.approx(reference.value, rel=1e-12, abs=1e-20)
        np.testing.assert_allclose(
            policy.weights.grad.numpy(), reference.gradient, rtol=1e-10, atol=1e-17,
        )


def test_failure_edges_keep_their_policy_probability_and_negative_gradient():
    world = OrderWorld(target_orders=(True, False))
    weights = np.zeros_like(initial_weights())
    policy = SoftmaxPolicy(weights)
    graph = compile_world(world, 3, prune_impossible=True)
    value = graph.expected_return(policy)
    value.backward()
    assert graph.node_count == 4
    assert graph.pruned_edges == 27
    assert float(value.detach()) == pytest.approx(0.001, rel=1e-13)
    assert float(policy.weights.grad[-1, -1]) == pytest.approx(-0.0003, rel=1e-13)


def test_pruning_preserves_later_actions_that_can_spoil_a_success():
    world = OrderWorld(target_orders=(False, False))
    policy = SoftmaxPolicy(np.zeros_like(initial_weights()))
    graph = compile_world(world, 1, prune_impossible=True)
    assert world.reward(world.initial_state()) == 1
    assert float(graph.expected_return(policy).detach()) == pytest.approx(0.8)


def test_pruning_reduces_storage_without_changing_the_fixed_deadline():
    world = OrderWorld()
    full = compile_world(world, 8)
    pruned = compile_world(world, 8, prune_impossible=True)
    assert len(pruned.layers) == len(full.layers) == 9
    assert pruned.node_count < full.node_count
    assert pruned.feasibility_evaluations > 0
    assert pruned.pruned_edges > 0
    assert full.feasibility_evaluations == full.pruned_edges == 0


def test_impossible_root_has_no_stored_states_and_a_valid_zero_gradient():
    graph = compile_world(OrderWorld(), 6, prune_impossible=True)
    policy = SoftmaxPolicy(initial_weights())
    value = graph.expected_return(policy)
    value.backward()
    assert graph.node_count == graph.edge_count == 0
    assert graph.feasibility_evaluations == 1
    assert float(value.detach()) == 0
    np.testing.assert_array_equal(policy.weights.grad.numpy(), np.zeros_like(initial_weights()))
