from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from backworld.environment import OrderWorld
from backworld.estimators import ExactSuffix
from backworld.importance import FeasibleImportance
from backworld.policy import action_score, observation, probabilities, validate_weights


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


class OracleLimitError(ValueError):
    def __init__(self, limit: str, nodes: int, elapsed_seconds: float):
        super().__init__(f"moment oracle exceeds {limit}")
        self.limit = limit
        self.nodes = nodes
        self.elapsed_seconds = elapsed_seconds


@dataclass(frozen=True)
class ImportanceMoments:
    value: float
    mean: np.ndarray
    value_second_moment: float
    gradient_second_moment: float
    proposal_success_probability: float
    layered_states: int
    policy_evaluations: int
    transitions: int
    feasibility_evaluations: int

    @property
    def value_variance(self) -> float:
        return max(0.0, self.value_second_moment - self.value ** 2)

    @property
    def variance_trace(self) -> float:
        return max(0.0, self.gradient_second_moment - float(np.sum(self.mean ** 2)))


def exact_forward(world: OrderWorld, weights: np.ndarray, horizon: int) -> OracleResult:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    validate_weights(weights, world)
    frontier = {world.initial_state(): (1.0, np.zeros_like(weights))}
    state_evaluations = 0
    for elapsed in range(horizon):
        following = {}
        for state, (mass, derivative) in frontier.items():
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            state_evaluations += 1
            for action in range(world.action_count):
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
    return OracleResult(value, gradient, state_evaluations, state_evaluations * world.action_count)


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
            for action in range(world.action_count):
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


def exact_importance_moments(
    world: OrderWorld, weights: np.ndarray, horizon: int,
    max_nodes: int = 5_000, max_seconds: float = 10.0,
) -> ImportanceMoments:
    if max_nodes < 1 or not isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_nodes and finite max_seconds must be positive")
    started = perf_counter()
    sampler = FeasibleImportance(world, weights, horizon)
    weights = sampler.weights
    frontier = {
        world.initial_state(): (
            1.0, 1.0, np.zeros_like(weights), 1.0, np.zeros_like(weights), 0.0,
        ),
    }
    nodes = 1

    def check_deadline(count):
        elapsed = perf_counter() - started
        if elapsed > max_seconds:
            raise OracleLimitError("max_seconds", count, elapsed)

    for elapsed in range(horizon):
        following = {}
        for state, (mass, first, derivative, second, weighted_score, squared_score) in frontier.items():
            check_deadline(nodes + len(following))
            proposal = sampler.proposal(state, horizon - elapsed)
            if proposal is None:
                continue
            for index, action in enumerate(proposal.actions):
                p = proposal.policy_probabilities[action]
                q = proposal.probabilities[index]
                second_factor = p * (p / q)
                if p == 0 or second_factor == 0:
                    raise FloatingPointError("positive importance moment factor underflow")
                score = action_score(proposal.features, proposal.policy_probabilities, action)
                values = (
                    q * mass,
                    p * first,
                    p * (derivative + first * score),
                    second_factor * second,
                    second_factor * (weighted_score + second * score),
                    second_factor * (
                        squared_score + 2 * np.sum(score * weighted_score)
                        + second * np.sum(score ** 2)
                    ),
                )
                if values[0] == 0 or values[1] == 0 or values[3] == 0:
                    raise FloatingPointError("positive importance prefix moment underflow")
                destination = proposal.destinations[index]
                if destination in following:
                    following[destination] = tuple(a + b for a, b in zip(following[destination], values))
                else:
                    if nodes + len(following) + 1 > max_nodes:
                        raise OracleLimitError("max_nodes", nodes + len(following), perf_counter() - started)
                    following[destination] = values
        nodes += len(following)
        frontier = following

    check_deadline(nodes)
    value = value_second = gradient_second = success_probability = 0.0
    mean = np.zeros_like(weights)
    for state, (mass, first, derivative, second, weighted_score, squared_score) in frontier.items():
        reward = world.reward(state)
        value += reward * first
        mean += reward * derivative
        value_second += reward ** 2 * second
        gradient_second += reward ** 2 * squared_score
        success_probability += reward * mass
    if not np.isfinite(mean).all() or not np.isfinite([value, value_second, gradient_second, success_probability]).all():
        raise FloatingPointError("nonfinite importance moments")
    return ImportanceMoments(
        float(value), mean, float(value_second), float(gradient_second),
        float(success_probability), nodes, sampler.policy_evaluations,
        sampler.transitions, len(sampler.bounds),
    )
