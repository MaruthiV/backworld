from dataclasses import dataclass

import numpy as np

from backworld.environment import ACTION_COUNT, OrderWorld
from backworld.estimators import ExactSuffix
from backworld.policy import action_score, observation, probabilities


@dataclass(frozen=True)
class OracleResult:
    value: float
    gradient: np.ndarray
    state_evaluations: int
    transitions: int


@dataclass(frozen=True)
class GradientMoments:
    mean: np.ndarray
    variance_trace: float


def exact_forward(world: OrderWorld, weights: np.ndarray, horizon: int) -> OracleResult:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    frontier = {world.initial_state(): (1.0, np.zeros_like(weights))}
    state_evaluations = 0
    for elapsed in range(horizon):
        following = {}
        for state, (mass, derivative) in frontier.items():
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            state_evaluations += 1
            for action in range(ACTION_COUNT):
                destination = world.step(state, action)
                next_mass = mass * probs[action]
                next_derivative = probs[action] * (
                    derivative + mass * action_score(features, probs, action)
                )
                if destination in following:
                    old_mass, old_derivative = following[destination]
                    following[destination] = (
                        old_mass + next_mass, old_derivative + next_derivative,
                    )
                else:
                    following[destination] = (next_mass, next_derivative)
        frontier = following

    value = 0.0
    gradient = np.zeros_like(weights)
    for state, (mass, derivative) in frontier.items():
        reward = world.reward(state)
        value += mass * reward
        gradient += derivative * reward
    return OracleResult(value, gradient, state_evaluations, state_evaluations * ACTION_COUNT)


def exact_suffix_moments(
    world: OrderWorld, weights: np.ndarray, horizon: int,
    exact_steps: int, baseline: float = 0.0,
) -> GradientMoments:
    estimator = ExactSuffix(world, weights, horizon, exact_steps, baseline)
    frontier = {world.initial_state(): (1.0, np.zeros_like(weights), 0.0)}
    for elapsed in range(horizon - exact_steps):
        following = {}
        for state, (mass, score_sum, score_squared) in frontier.items():
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            for action in range(ACTION_COUNT):
                score = action_score(features, probs, action)
                next_mass = probs[action] * mass
                next_sum = probs[action] * (score_sum + mass * score)
                next_squared = probs[action] * (
                    score_squared + 2 * np.sum(score * score_sum) + mass * np.sum(score ** 2)
                )
                destination = world.step(state, action)
                old_mass, old_sum, old_squared = following.get(
                    destination, (0.0, np.zeros_like(weights), 0.0),
                )
                following[destination] = (
                    old_mass + next_mass, old_sum + next_sum, old_squared + next_squared,
                )
        frontier = following

    mean = np.zeros_like(weights)
    second_moment = 0.0
    for state, (mass, score_sum, score_squared) in frontier.items():
        value, derivative = estimator.continuation(state, exact_steps)
        centered = value - baseline
        mean += centered * score_sum + mass * derivative
        second_moment += (
            centered ** 2 * score_squared
            + 2 * centered * np.sum(derivative * score_sum)
            + mass * np.sum(derivative ** 2)
        )
    return GradientMoments(mean, max(0.0, float(second_moment - np.sum(mean ** 2))))
