from itertools import product

import numpy as np
import pytest
import torch

from backworld import OrderWorld, SoftmaxPolicy, compile_world
from backworld.environment import ACTION_COUNT
from backworld.estimators import ExactSuffix
from backworld.oracle import exact_forward, exact_suffix_moments
from backworld.policy import action_score, initial_weights, observation, probabilities


torch.set_num_threads(1)


@pytest.mark.parametrize("world,horizon", [
    (OrderWorld(), 8),
    (OrderWorld(shared_inventory=False, replenishments=0), 7),
    (OrderWorld(target_orders=(True, False)), 3),
    (OrderWorld(), 2),
    (OrderWorld(), 0),
])
def test_backward_graph_matches_independent_forward_derivatives(world, horizon):
    weights = initial_weights(19)
    reference = exact_forward(world, weights, horizon)
    graph = compile_world(world, horizon)
    policy = SoftmaxPolicy(weights)
    value = graph.expected_return(policy)
    value.backward()
    assert float(value.detach()) == pytest.approx(reference.value, abs=1e-13)
    np.testing.assert_allclose(policy.weights.grad.numpy(), reference.gradient, atol=1e-12)


def test_one_order_three_step_success_has_an_independent_product_formula():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights(7)
    state = world.initial_state()
    expected = 1.0
    for remaining, action in zip([3, 2, 1], [0, 1, 2]):
        expected *= probabilities(weights, observation(world, state, remaining, 3))[action]
        state = world.step(state, action)
    assert exact_forward(world, weights, 3).value == pytest.approx(expected, rel=1e-13)


def test_forward_gradient_matches_numerical_derivatives():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights(31)
    reference = exact_forward(world, weights, 4)
    epsilon = 1e-5
    for coordinate in [(0, 15), (1, 0), (2, 4), (4, 11), (9, 9)]:
        plus = weights.copy()
        minus = weights.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        numerical = (
            exact_forward(world, plus, 4).value - exact_forward(world, minus, 4).value
        ) / (2 * epsilon)
        assert reference.gradient[coordinate] == pytest.approx(numerical, abs=1e-9)


@pytest.mark.parametrize("exact_steps", [0, 1, 2, 3])
@pytest.mark.parametrize("baseline", [0.0, 0.37])
def test_exhaustive_prefix_average_preserves_the_full_gradient(exact_steps, baseline):
    world = OrderWorld(target_orders=(True, False))
    horizon = 3
    weights = initial_weights(7)
    estimator = ExactSuffix(world, weights, horizon, exact_steps, baseline)
    total_gradient = np.zeros_like(weights)
    total_probability = 0.0
    second_moment = 0.0
    for actions in product(range(ACTION_COUNT), repeat=horizon - exact_steps):
        state = world.initial_state()
        mass = 1.0
        score = np.zeros_like(weights)
        for elapsed, action in enumerate(actions):
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            mass *= probs[action]
            score += action_score(features, probs, action)
            state = world.step(state, action)
        value, derivative = estimator.continuation(state, exact_steps)
        sample_gradient = (value - baseline) * score + derivative
        total_gradient += mass * sample_gradient
        second_moment += mass * np.sum(sample_gradient ** 2)
        total_probability += mass
    assert total_probability == pytest.approx(1.0, abs=1e-12)
    reference = exact_forward(world, weights, horizon)
    np.testing.assert_allclose(total_gradient, reference.gradient, atol=1e-12)
    moments = exact_suffix_moments(world, weights, horizon, exact_steps, baseline)
    np.testing.assert_allclose(moments.mean, total_gradient, atol=1e-12)
    expected_variance = max(0.0, second_moment - np.sum(total_gradient ** 2))
    assert moments.variance_trace == pytest.approx(expected_variance, abs=1e-12)


def test_full_suffix_is_exact_and_zero_suffix_is_sampled_reinforce():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights(7)
    reference = exact_forward(world, weights, 3)
    full = ExactSuffix(world, weights, 3, 3)
    value, gradient = full.estimate(np.random.default_rng(8))
    assert value == pytest.approx(reference.value)
    np.testing.assert_allclose(gradient, reference.gradient, atol=1e-12)
    sampled = ExactSuffix(world, weights, 3, 0)
    value, _ = sampled.estimate(np.random.default_rng(8))
    assert value in (0.0, 1.0)


def test_remaining_time_is_part_of_the_policy_input():
    world = OrderWorld()
    state = world.initial_state()
    first = observation(world, state, 8, 8)
    later = observation(world, state, 3, 8)
    assert first[11] == 1.0
    assert later[11] == 3 / 8
    assert not np.array_equal(first, later)


def test_compiled_topology_recomputes_values_after_a_policy_update():
    world = OrderWorld(target_orders=(True, False))
    graph = compile_world(world, 3)
    policy = SoftmaxPolicy(initial_weights(7))
    previous = float(graph.expected_return(policy).detach())
    with torch.no_grad():
        policy.weights[0, 15] += 0.5
    updated = graph.expected_return(policy)
    updated.backward()
    reference = exact_forward(world, policy.weights.detach().numpy(), 3)
    assert float(updated.detach()) != pytest.approx(previous)
    assert float(updated.detach()) == pytest.approx(reference.value)
    np.testing.assert_allclose(policy.weights.grad.numpy(), reference.gradient, atol=1e-12)


def test_suffix_cache_is_scoped_to_a_frozen_weight_snapshot():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights(7)
    estimator = ExactSuffix(world, weights, 3, 3)
    before = estimator.estimate(np.random.default_rng(0))
    weights[:] = 10.0
    after = estimator.estimate(np.random.default_rng(0))
    np.testing.assert_array_equal(before[1], after[1])
    assert not estimator.weights.flags.writeable


def test_graph_size_limit_fails_before_unbounded_expansion():
    with pytest.raises(ValueError, match="max_nodes"):
        compile_world(OrderWorld(), 100, max_nodes=2)


@pytest.mark.parametrize("depth", [-1, 4])
def test_invalid_suffix_depth_is_rejected(depth):
    with pytest.raises(ValueError):
        ExactSuffix(OrderWorld(), initial_weights(), 3, depth)


def test_exact_variance_decreases_when_more_of_the_future_is_integrated():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights(19)
    variances = [exact_suffix_moments(world, weights, 4, depth).variance_trace for depth in range(5)]
    assert all(after <= before + 1e-12 for before, after in zip(variances, variances[1:]))
    assert variances[0] > 0
    assert variances[-1] == pytest.approx(0.0, abs=1e-14)
