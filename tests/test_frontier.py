from itertools import count, product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from backworld import OrderWorld, compile_world
from backworld.feasibility import completion_lower_bound
from backworld.frontier import (
    FrontierLimitError, build_frontier_truth, frontier_moments, nonnegative,
    residual_variance, search_exact_sets,
)
from backworld.importance import FeasibleImportance
from backworld.oracle import exact_forward, exact_importance_moments
from backworld.policy import action_score, initial_weights, observation, probabilities


def enumerate_prefixes(world, weights, horizon, depth):
    grouped = {}
    for actions in product(range(world.action_count), repeat=depth):
        state, mass, score = world.initial_state(), 1.0, np.zeros(weights.size)
        for elapsed, action in enumerate(actions):
            features = observation(world, state, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            mass *= probs[action]
            score += action_score(features, probs, action).ravel()
            state = world.step(state, action)
        grouped.setdefault(state, []).append((mass, score))
    return grouped


def enumerate_tails(world, weights, horizon, depth, state, proposal):
    sampler = FeasibleImportance(world, weights, horizon)
    outcomes = []
    for actions in product(range(world.action_count), repeat=horizon - depth):
        current, q_mass, weight, score = state, 1.0, 1.0, np.zeros(weights.size)
        for elapsed, action in enumerate(actions, start=depth):
            features = observation(world, current, horizon - elapsed, horizon)
            probs = probabilities(weights, features)
            q = probs[action]
            if proposal == "feasible":
                guided = sampler.proposal(current, horizon - elapsed)
                if guided is None or action not in guided.actions:
                    q_mass = 0.0
                    break
                q = guided.probabilities[guided.actions.index(action)]
            q_mass *= q
            weight *= probs[action] / q
            score += action_score(features, probs, action).ravel()
            current = world.step(current, action)
        if q_mass:
            outcomes.append((q_mass, weight * world.reward(current), score))
    return outcomes


@pytest.mark.parametrize("proposal", ["original", "feasible"])
@pytest.mark.parametrize("world", [
    OrderWorld(target_orders=(True,), replenishments=0),
    OrderWorld(target_orders=(True, False), replenishments=0),
])
def test_frontier_moments_and_residual_estimator_match_flat_paths(world, proposal):
    horizon, depth = 4, 2
    weights = initial_weights(19, world)
    graph = compile_world(world, horizon, prune_impossible=True)
    truth = build_frontier_truth(world, weights, graph, depth)
    moments = frontier_moments(truth, proposal)
    prefixes = enumerate_prefixes(world, weights, horizon, depth)
    tails = []
    merged_different_scores = False
    for index, state in enumerate(truth.states):
        paths = prefixes[state]
        mass = sum(probability for probability, _ in paths)
        derivative = sum(probability * score for probability, score in paths)
        np.testing.assert_allclose(truth.masses[index], mass, rtol=1e-13)
        np.testing.assert_allclose(truth.mass_gradients[index], derivative, rtol=1e-12, atol=1e-16)
        merged_different_scores |= any(not np.allclose(paths[0][1], score) for _, score in paths[1:])
        outcomes = enumerate_tails(world, weights, horizon, depth, state, proposal)
        tails.append(outcomes)
        second = sum(q * value ** 2 for q, value, _ in outcomes)
        weighted = sum((q * value ** 2 * score for q, value, score in outcomes), np.zeros(weights.size))
        squared = sum(q * value ** 2 * np.dot(score, score) for q, value, score in outcomes)
        np.testing.assert_allclose(moments.second_values[index], second, rtol=1e-12, atol=1e-17)
        np.testing.assert_allclose(moments.weighted_scores[index], weighted, rtol=1e-12, atol=1e-17)
        np.testing.assert_allclose(moments.squared_scores[index], squared, rtol=1e-12, atol=1e-17)
    assert merged_different_scores
    reference = exact_forward(world, weights, horizon)
    np.testing.assert_allclose(truth.contributions.sum(axis=0), reference.gradient.ravel(), rtol=1e-12, atol=1e-16)
    wrong_gradient = (truth.masses[:, None] * truth.gradients).sum(axis=0)
    assert np.linalg.norm(wrong_gradient - reference.gradient.ravel()) > 1e-5
    for selected in ((), (0,), tuple(range(len(truth.states)))):
        residual = [index for index in range(len(truth.states)) if index not in selected]
        total_mass = sum(truth.masses[index] for index in residual)
        mean, second = np.zeros(weights.size), 0.0
        for index in residual:
            mu = truth.mass_gradients[index] / truth.masses[index]
            for q, value, score in tails[index]:
                contribution = total_mass * value * (mu + score)
                probability = truth.masses[index] / total_mass * q
                mean += probability * contribution
                second += probability * np.dot(contribution, contribution)
        complete_mean = mean + truth.contributions[list(selected)].sum(axis=0)
        np.testing.assert_allclose(complete_mean, reference.gradient.ravel(), rtol=1e-12, atol=1e-16)
        expected_variance = second - np.dot(mean, mean)
        assert residual_variance(truth, moments, selected) == pytest.approx(expected_variance, rel=1e-12, abs=1e-17)
    assert residual_variance(truth, moments, tuple(range(len(truth.states)))) == 0


def test_depth_zero_matches_independent_forward_importance_moments():
    world = OrderWorld(replenishments=0, target_orders=(True, False))
    weights = initial_weights(31, world)
    graph = compile_world(world, 4, prune_impossible=True)
    truth = build_frontier_truth(world, weights, graph, 0)
    moments = frontier_moments(truth, "feasible")
    reference = exact_importance_moments(world, weights, 4)
    assert residual_variance(truth, moments, ()) == pytest.approx(reference.variance_trace, rel=1e-12)


def test_shared_closures_and_best_subsets_match_independent_sets():
    world = OrderWorld(replenishments=0, target_orders=(True,))
    weights = initial_weights(7, world)
    horizon, depth = 4, 2
    graph = compile_world(world, horizon, prune_impossible=True)
    truth = build_frontier_truth(world, weights, graph, depth)
    moments = frontier_moments(truth, "original")
    prefixes = {(state, horizon - elapsed) for elapsed, layer in enumerate(graph.layers[:depth + 1]) for state in layer}
    closures = []
    for state in truth.states:
        discovered = {(state, horizon - depth)}
        queue = list(discovered)
        while queue:
            current, remaining = queue.pop()
            if remaining == 0:
                continue
            for action in range(world.action_count):
                following = world.step(current, action)
                bound = completion_lower_bound(world, following)
                key = (following, remaining - 1)
                if bound is not None and bound <= remaining - 1 and key not in discovered:
                    discovered.add(key)
                    queue.append(key)
        closures.append(discovered)
    budgets = tuple(range(len(prefixes) - 1, graph.node_count + 1))
    search = search_exact_sets(truth, moments, budgets)
    options = []
    for choices in product((False, True), repeat=len(truth.states)):
        selected = tuple(index for index, choice in enumerate(choices) if choice)
        keys = prefixes.union(*(closures[index] for index in selected))
        assert truth.exact_nodes(selected) == len(keys)
        options.append((len(keys), residual_variance(truth, moments, selected)))
    assert truth.exact_nodes(tuple(range(len(truth.states)))) == graph.node_count
    assert len(set.union(*closures)) < sum(map(len, closures))
    for record in search["budgets"]:
        feasible = [variance for nodes, variance in options if nodes <= record["budget_nodes"]]
        if not feasible:
            assert record["status"] == "prefix_budget_infeasible"
        else:
            assert record["ideal"]["variance_trace"] == min(feasible)
            assert record["probability_ordered"]["exact_nodes"] <= record["budget_nodes"]


def test_numeric_limits_frozen_weights_and_empty_graph(monkeypatch):
    world = OrderWorld(replenishments=0, target_orders=(True,))
    weights = initial_weights(7, world)
    graph = compile_world(world, 4, prune_impossible=True)
    truth = build_frontier_truth(world, weights, graph, 2)
    weights[:] = 100
    assert not np.array_equal(truth.weights, weights)
    assert not truth.weights.flags.writeable
    with pytest.raises(FrontierLimitError, match="max_frontier"):
        search_exact_sets(truth, frontier_moments(truth, "original"), (graph.node_count,), max_frontier=0)
    clock = count(0.0, 1.0)
    monkeypatch.setattr("backworld.frontier.perf_counter", lambda: next(clock))
    with pytest.raises(FrontierLimitError, match="moment_seconds"):
        frontier_moments(truth, "original", max_seconds=0.5)
    with pytest.raises(FrontierLimitError, match="search_seconds"):
        search_exact_sets(truth, frontier_moments(truth, "original", max_seconds=1000), (10,), max_seconds=0.5)
    with pytest.raises(FloatingPointError, match="negative"):
        nonnegative(-1e-10, 1e-10)
    assert nonnegative(-1e-25, 1e-10) == 0
    with pytest.raises(ValueError, match="depth"):
        build_frontier_truth(world, weights, graph, 5)
    empty_graph = compile_world(world, 0, prune_impossible=True)
    empty = build_frontier_truth(world, weights, empty_graph, 0)
    assert empty.root_value == 0
    assert residual_variance(empty, frontier_moments(empty, "original"), ()) == 0


def test_ceiling_supervisor_stays_lightweight_and_reports_refusals(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    light = subprocess.run([
        sys.executable, "-c", "import sys; import experiments.frontier_ceiling; assert 'torch' not in sys.modules and 'numpy' not in sys.modules",
    ], cwd=root, env=env, capture_output=True, text=True, timeout=10)
    assert light.returncode == 0, light.stderr
    for max_frontier, expected in ((18, "complete"), (0, "partial_resource_limit")):
        output = tmp_path / f"ceiling-{max_frontier}.json"
        completed = subprocess.run([
            sys.executable, "-m", "experiments.frontier_ceiling", "--orders", "1",
            "--variants", "separate", "--seeds", "7", "--max-frontier", str(max_frontier),
            "--output", str(output),
        ], cwd=root, env=env, capture_output=True, text=True, timeout=20)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = json.loads(output.read_text())
        case, = report["cases"]
        assert case["status"] == expected
        assert set(case["proposals"]) == {"original", "feasible"}
        for proposal in case["proposals"].values():
            assert proposal["status"] == ("complete" if max_frontier else "resource_limit")
            if not max_frontier:
                assert "budgets" not in proposal
