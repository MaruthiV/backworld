from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from backworld.environment import OrderState, OrderWorld
from backworld.graph import CompiledWorld
from backworld.importance import FeasibleImportance
from backworld.policy import action_score, probabilities, validate_weights


class FrontierLimitError(ValueError):
    def __init__(self, limit: str, elapsed_seconds: float):
        super().__init__(f"frontier diagnostic exceeds {limit}")
        self.limit = limit
        self.elapsed_seconds = elapsed_seconds


def nonnegative(value: float, scale: float) -> float:
    if not np.isfinite([value, scale]).all() or scale < 0:
        raise FloatingPointError("invalid second moment or variance")
    tolerance = 128 * np.finfo(np.float64).eps * max(scale, np.finfo(np.float64).tiny)
    if value < -tolerance:
        raise FloatingPointError("negative second moment or variance exceeds roundoff")
    return max(0.0, float(value))


@dataclass(frozen=True)
class FrontierTruth:
    world: OrderWorld
    graph: CompiledWorld
    weights: np.ndarray
    depth: int
    probabilities: tuple[np.ndarray, ...]
    masses: np.ndarray
    mass_gradients: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    closures: tuple[int, ...]
    prefix_bits: int
    root_value: float
    root_gradient: np.ndarray

    @property
    def states(self) -> tuple[OrderState, ...]:
        return self.graph.layers[self.depth]

    @property
    def contributions(self) -> np.ndarray:
        return self.mass_gradients * self.values[:, None] + self.masses[:, None] * self.gradients

    def exact_nodes(self, selected: tuple[int, ...]) -> int:
        bits = self.prefix_bits
        for index in selected:
            bits |= self.closures[index]
        return bits.bit_count()


def build_frontier_truth(
    world: OrderWorld, weights: np.ndarray, graph: CompiledWorld, depth: int,
) -> FrontierTruth:
    horizon = len(graph.transitions)
    if not 0 <= depth <= horizon:
        raise ValueError("cut depth must lie within the horizon")
    weights = np.array(weights, dtype=np.float64, copy=True)
    validate_weights(weights, world)
    weights.flags.writeable = False
    dimension = weights.size
    layer_probs = []
    for features in graph.features:
        rows = np.empty((len(features), world.action_count))
        for index, feature in enumerate(features.numpy()):
            rows[index] = probabilities(weights, feature)
        if not np.isfinite(rows).all() or np.any(rows <= 0):
            raise FloatingPointError("policy support is not numerically representable")
        layer_probs.append(rows)

    masses = np.ones(len(graph.layers[0]))
    derivatives = np.zeros((len(masses), dimension))
    for elapsed in range(depth):
        next_mass = np.zeros(len(graph.layers[elapsed + 1]))
        next_derivative = np.zeros((len(next_mass), dimension))
        for index, targets in enumerate(graph.transitions[elapsed].numpy()):
            features = graph.features[elapsed][index].numpy()
            probs = layer_probs[elapsed][index]
            for action, destination in enumerate(targets):
                if destination < 0:
                    continue
                score = action_score(features, probs, action).ravel()
                next_mass[destination] += probs[action] * masses[index]
                next_derivative[destination] += probs[action] * (
                    derivatives[index] + masses[index] * score
                )
        masses, derivatives = next_mass, next_derivative
    if np.any(masses <= 0) or not np.isfinite(derivatives).all():
        raise FloatingPointError("frontier mass or derivative is not representable")

    offsets = np.cumsum([0, *map(len, graph.layers)])
    values = graph.rewards.numpy().copy()
    gradients = np.zeros((len(values), dimension))
    closures = [1 << (int(offsets[-2]) + index) for index in range(len(values))]
    cut_values, cut_gradients, cut_closures = values, gradients, tuple(closures)
    for elapsed in range(horizon - 1, -1, -1):
        current_values = np.zeros(len(graph.layers[elapsed]))
        current_gradients = np.zeros((len(current_values), dimension))
        current_closures = []
        for index, targets in enumerate(graph.transitions[elapsed].numpy()):
            features = graph.features[elapsed][index].numpy()
            probs = layer_probs[elapsed][index]
            bits = 1 << (int(offsets[elapsed]) + index)
            for action, destination in enumerate(targets):
                if destination < 0:
                    continue
                score = action_score(features, probs, action).ravel()
                current_values[index] += probs[action] * values[destination]
                current_gradients[index] += probs[action] * (
                    gradients[destination] + values[destination] * score
                )
                bits |= closures[destination]
            current_closures.append(bits)
        values, gradients, closures = current_values, current_gradients, current_closures
        if elapsed == depth:
            cut_values, cut_gradients, cut_closures = values, gradients, tuple(closures)
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise FloatingPointError("nonfinite exact suffix calculation")
    return FrontierTruth(
        world, graph, weights, depth, tuple(layer_probs), masses, derivatives,
        cut_values, cut_gradients, cut_closures,
        (1 << int(offsets[depth + 1])) - 1,
        float(values[0]) if len(values) else 0.0,
        gradients[0] if len(gradients) else np.zeros(dimension),
    )


@dataclass(frozen=True)
class FrontierMoments:
    proposal: str
    second_values: np.ndarray
    weighted_scores: np.ndarray
    squared_scores: np.ndarray
    full_score_second_moments: np.ndarray
    policy_evaluations: int


def frontier_moments(
    truth: FrontierTruth, proposal: str, max_seconds: float = 10.0,
) -> FrontierMoments:
    if proposal not in ("original", "feasible"):
        raise ValueError("unknown suffix proposal")
    if not isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("moment time limit must be finite and positive")
    started = perf_counter()
    world, graph = truth.world, truth.graph
    horizon = len(graph.transitions)
    sampler = FeasibleImportance(world, truth.weights, horizon) if proposal == "feasible" else None
    dimension = truth.weights.size
    second = graph.rewards.numpy() ** 2
    weighted = np.zeros((len(second), dimension))
    squared = np.zeros(len(second))
    for elapsed in range(horizon - 1, truth.depth - 1, -1):
        next_second = np.zeros(len(graph.layers[elapsed]))
        next_weighted = np.zeros((len(next_second), dimension))
        next_squared = np.zeros(len(next_second))
        for index, targets in enumerate(graph.transitions[elapsed].numpy()):
            elapsed_seconds = perf_counter() - started
            if elapsed_seconds > max_seconds:
                raise FrontierLimitError("moment_seconds", elapsed_seconds)
            features = graph.features[elapsed][index].numpy()
            probs = truth.probabilities[elapsed][index]
            q = probs
            if sampler is not None:
                guided = sampler.proposal(graph.layers[elapsed][index], horizon - elapsed)
                q = np.zeros_like(probs)
                if guided is not None:
                    q[list(guided.actions)] = guided.probabilities
            for action, destination in enumerate(targets):
                if destination < 0:
                    continue
                if q[action] <= 0:
                    raise FloatingPointError("suffix proposal removed a retained transition")
                factor = probs[action] * (probs[action] / q[action])
                if factor == 0 and second[destination] > 0:
                    raise FloatingPointError("positive suffix second moment underflow")
                score = action_score(features, probs, action).ravel()
                next_second[index] += factor * second[destination]
                next_weighted[index] += factor * (
                    weighted[destination] + second[destination] * score
                )
                next_squared[index] += factor * (
                    squared[destination] + 2 * np.dot(score, weighted[destination])
                    + second[destination] * np.dot(score, score)
                )
        second, weighted, squared = next_second, next_weighted, next_squared
    means = np.divide(
        truth.mass_gradients, truth.masses[:, None],
        out=np.zeros_like(truth.mass_gradients), where=truth.masses[:, None] > 0,
    )
    first = second * np.sum(means ** 2, axis=1)
    cross = 2 * np.sum(means * weighted, axis=1)
    full = np.asarray([
        nonnegative(a + b + c, abs(a) + abs(b) + abs(c))
        for a, b, c in zip(first, cross, squared)
    ])
    if not np.isfinite(second).all() or not np.isfinite(weighted).all() or not np.isfinite(squared).all():
        raise FloatingPointError("nonfinite suffix moments")
    return FrontierMoments(
        proposal, second, weighted, squared, full,
        sampler.policy_evaluations if sampler is not None else 0,
    )


def residual_variance(truth: FrontierTruth, moments: FrontierMoments, selected: tuple[int, ...]) -> float:
    residual = np.ones(len(truth.states), dtype=bool)
    residual[list(selected)] = False
    mass = float(truth.masses[residual].sum())
    second = mass * float(np.sum(truth.masses[residual] * moments.full_score_second_moments[residual]))
    mean = truth.contributions[residual].sum(axis=0)
    squared_mean = float(np.dot(mean, mean))
    return nonnegative(second - squared_mean, second + squared_mean)


def selection_record(truth: FrontierTruth, moments: FrontierMoments, selected: tuple[int, ...]) -> dict:
    gradient = truth.contributions.sum(axis=0)
    norm_squared = float(np.dot(gradient, gradient))
    covered = truth.contributions[list(selected)].sum(axis=0)
    variance = residual_variance(truth, moments, selected)
    empty_variance = residual_variance(truth, moments, ())
    return {
        "selected": list(selected), "selected_roots": len(selected),
        "exact_nodes": truth.exact_nodes(selected),
        "independent_closure_nodes": sum(truth.closures[index].bit_count() for index in selected),
        "exact_node_fraction": truth.exact_nodes(selected) / max(1, truth.graph.node_count),
        "variance_trace": variance,
        "relative_variance": variance / norm_squared if norm_squared else None,
        "noise_reduction_fraction": 1 - variance / empty_variance if empty_variance else None,
        "gradient_projection_covered": float(np.dot(covered, gradient)) / norm_squared if norm_squared else None,
        "gradient_norm_ratio_covered": float(np.linalg.norm(covered)) / np.sqrt(norm_squared) if norm_squared else None,
    }


def search_exact_sets(
    truth: FrontierTruth, moments: FrontierMoments, budgets: tuple[int, ...],
    max_frontier: int = 18, max_seconds: float = 10.0,
) -> dict:
    if not budgets or min(budgets) < 0 or max_frontier < 0 or not isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("subset budgets and finite search limits must be valid")
    started = perf_counter()
    count = len(truth.states)
    if count > max_frontier:
        raise FrontierLimitError("max_frontier", perf_counter() - started)
    best = [None] * len(budgets)
    best_variances = [float("inf")] * len(budgets)
    contributions = truth.contributions
    for mask in range(1 << count):
        elapsed = perf_counter() - started
        if elapsed > max_seconds:
            raise FrontierLimitError("search_seconds", elapsed)
        bits = truth.prefix_bits
        selected = []
        remaining = mask
        while remaining:
            low = remaining & -remaining
            index = low.bit_length() - 1
            selected.append(index)
            bits |= truth.closures[index]
            remaining ^= low
        nodes = bits.bit_count()
        if nodes > max(budgets):
            continue
        residual = np.ones(count, dtype=bool)
        residual[selected] = False
        mass = float(truth.masses[residual].sum())
        second = mass * float(np.sum(truth.masses[residual] * moments.full_score_second_moments[residual]))
        mean = contributions[residual].sum(axis=0)
        squared_mean = float(np.dot(mean, mean))
        variance = nonnegative(second - squared_mean, second + squared_mean)
        for index, budget in enumerate(budgets):
            if nodes <= budget and (
                variance < best_variances[index]
                or (variance == best_variances[index] and nodes < truth.exact_nodes(best[index]))
            ):
                best[index], best_variances[index] = tuple(selected), variance
    records = []
    order = sorted(range(count), key=lambda index: (-truth.masses[index], index))
    for budget, selected in zip(budgets, best):
        greedy = ()
        if truth.exact_nodes(()) > budget:
            records.append({"budget_nodes": budget, "status": "prefix_budget_infeasible"})
            continue
        for index in order:
            trial = tuple(sorted((*greedy, index)))
            if truth.exact_nodes(trial) <= budget:
                greedy = trial
        records.append({
            "budget_nodes": budget, "status": "complete",
            "ideal": selection_record(truth, moments, selected),
            "probability_ordered": selection_record(truth, moments, greedy),
        })
    return {
        "status": "complete", "subsets_checked": 1 << count,
        "search_seconds": perf_counter() - started,
        "empty": selection_record(truth, moments, ()),
        "all_exact": selection_record(truth, moments, tuple(range(count))),
        "budgets": records,
    }
