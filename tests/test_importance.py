import numpy as np
import pytest

from backworld import OrderWorld
from backworld.importance import FeasibleImportance
from backworld.oracle import exact_forward
from backworld.policy import action_score, initial_weights, observation, probabilities


class FixedChoices:
    def __init__(self, choices):
        self.choices = iter(choices)

    def choice(self, count, p):
        choice = next(self.choices)
        assert 0 <= choice < count and p[choice] > 0
        return choice


def proposal_paths(estimator, state, remaining, mass=1.0, choices=(), actions=()):
    if remaining == 0:
        yield mass, choices, actions
        return
    proposal = estimator.proposal(state, remaining)
    if proposal is None:
        yield mass, choices, actions
        return
    for index, action in enumerate(proposal.actions):
        yield from proposal_paths(
            estimator, proposal.destinations[index], remaining - 1,
            mass * proposal.probabilities[index], choices + (index,), actions + (action,),
        )


def check_exhaustive_estimates(world, horizon):
    weights = initial_weights(19, world)
    sampler = FeasibleImportance(world, weights, horizon)
    reference = exact_forward(world, weights, horizon)
    total_mass = 0.0
    mean_value = 0.0
    mean_gradient = np.zeros_like(weights)
    second_value = second_gradient = 0.0
    expected_second_value = expected_second_gradient = 0.0
    for mass, choices, actions in proposal_paths(sampler, world.initial_state(), horizon):
        value, gradient = sampler.estimate(FixedChoices(choices))
        state = world.initial_state()
        policy_mass = 1.0
        score = np.zeros_like(weights)
        for elapsed, action in enumerate(actions):
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            policy_mass *= probs[action]
            score += action_score(features, probs, action)
            state = world.step(state, action)
        reward = world.reward(state) if len(actions) == horizon else 0.0
        corrected = policy_mass * reward / mass
        assert value == pytest.approx(corrected, rel=1e-12, abs=1e-20)
        np.testing.assert_allclose(gradient, corrected * score, atol=1e-17, rtol=1e-10)
        total_mass += mass
        mean_value += mass * value
        mean_gradient += mass * gradient
        second_value += mass * value ** 2
        second_gradient += mass * np.sum(gradient ** 2)
        original_squared_mass = (policy_mass * reward) ** 2 / mass
        expected_second_value += original_squared_mass
        expected_second_gradient += original_squared_mass * np.sum(score ** 2)
    assert total_mass == pytest.approx(1.0, abs=1e-12)
    assert mean_value == pytest.approx(reference.value, rel=1e-12, abs=1e-20)
    np.testing.assert_allclose(mean_gradient, reference.gradient, atol=1e-16, rtol=1e-10)
    assert second_value == pytest.approx(expected_second_value, rel=1e-12, abs=1e-25)
    assert second_gradient == pytest.approx(expected_second_gradient, rel=1e-12, abs=1e-25)
    assert second_gradient + 1e-16 >= np.sum(reference.gradient ** 2)


@pytest.mark.parametrize("world,horizon", [
    (OrderWorld(target_orders=(True, False)), 3),
    (OrderWorld(target_orders=(True, False)), 4),
    (OrderWorld(), 7),
    (OrderWorld(False, 1, 0), 6),
    (OrderWorld(target_orders=(False, False)), 2),
    (OrderWorld(target_orders=(False, False)), 0),
    (OrderWorld(), 0),
    (OrderWorld(True, 0, 2), 2),
    (OrderWorld(False, 0, 2, (False, True)), 3),
])
def test_exact_proposal_averages_and_moments_preserve_the_original_policy(world, horizon):
    check_exhaustive_estimates(world, horizon)


def test_forced_success_keeps_the_excluded_waiting_actions_negative_gradient():
    world = OrderWorld(target_orders=(True, False))
    sampler = FeasibleImportance(world, np.zeros_like(initial_weights()), 3)
    value, gradient = sampler.estimate(np.random.default_rng(7))
    assert value == pytest.approx(0.001, rel=1e-13)
    assert gradient[-1, -1] == pytest.approx(-0.0003, rel=1e-13)
    assert sampler.sampled_steps == 3
    assert sampler.transitions == 30


def test_a_weaker_bound_can_return_failures_without_bias(monkeypatch):
    monkeypatch.setattr("backworld.importance.completion_lower_bound", lambda world, state: 0)
    world = OrderWorld(target_orders=(True, False))
    check_exhaustive_estimates(world, 3)
    sampler = FeasibleImportance(world, initial_weights(), 3)
    value, gradient = sampler.estimate(FixedChoices((9, 9, 9)))
    assert value == 0
    np.testing.assert_array_equal(gradient, np.zeros_like(initial_weights()))


def test_policy_snapshot_and_cached_proposals_survive_caller_weight_mutation():
    world = OrderWorld(target_orders=(True, False))
    weights = initial_weights()
    sampler = FeasibleImportance(world, weights, 3)
    before = sampler.estimate(np.random.default_rng(1))
    evaluations = sampler.policy_evaluations
    transitions = sampler.transitions
    weights[:] = 100
    after = sampler.estimate(np.random.default_rng(1))
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert sampler.policy_evaluations == evaluations
    assert sampler.transitions == transitions
    assert not sampler.weights.flags.writeable


def test_impossible_root_returns_zero_without_sampling():
    sampler = FeasibleImportance(OrderWorld(), initial_weights(), 2)
    value, gradient = sampler.estimate(FixedChoices(()))
    assert value == 0
    assert sampler.transitions == sampler.policy_evaluations == sampler.sampled_steps == 0
    np.testing.assert_array_equal(gradient, np.zeros_like(initial_weights()))


def test_positive_return_underflow_is_reported_instead_of_returning_zero():
    world = OrderWorld(target_orders=(True, False))
    weights = np.zeros_like(initial_weights())
    weights[-1, -1] = 400
    sampler = FeasibleImportance(world, weights, 3)
    with pytest.raises(FloatingPointError, match="weight underflow"):
        sampler.estimate(np.random.default_rng(0))
    assert sampler.last_log_weight < -1000


def test_losing_proposal_support_to_underflow_is_reported():
    world = OrderWorld(target_orders=(True, False))
    weights = np.zeros_like(initial_weights())
    weights[-1, -1] = 1000
    sampler = FeasibleImportance(world, weights, 4)
    with pytest.raises(FloatingPointError, match="success support"):
        sampler.estimate(np.random.default_rng(0))


def test_invalid_horizon_and_mismatched_weights_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        FeasibleImportance(OrderWorld(), initial_weights(), -1)
    with pytest.raises(ValueError, match="dimensions"):
        FeasibleImportance(OrderWorld(target_orders=(True,)), initial_weights(), 3)
