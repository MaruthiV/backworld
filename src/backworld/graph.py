from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
import torch

from backworld.environment import OrderState, OrderWorld
from backworld.feasibility import completion_lower_bound
from backworld.policy import SoftmaxPolicy, observation


class GraphLimitError(ValueError):
    def __init__(self, limit: str, nodes: int, elapsed_seconds: float):
        super().__init__(f"reachable graph exceeds {limit}")
        self.limit = limit
        self.nodes = nodes
        self.elapsed_seconds = elapsed_seconds


@dataclass(frozen=True)
class CompiledWorld:
    layers: tuple[tuple[OrderState, ...], ...]
    transitions: tuple[torch.Tensor, ...]
    features: tuple[torch.Tensor, ...]
    rewards: torch.Tensor
    pruned_edges: int = 0
    feasibility_evaluations: int = 0

    @property
    def node_count(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def edge_count(self) -> int:
        return sum(transition.numel() for transition in self.transitions)

    def expected_return(self, policy: SoftmaxPolicy) -> torch.Tensor:
        weights = policy.weights
        if not self.layers[0]:
            return weights.sum() * 0
        values = self.rewards.to(weights)
        for features, targets in zip(reversed(self.features), reversed(self.transitions)):
            probs = policy(features.to(weights))
            values = torch.cat((values, weights.new_zeros(1)))
            values = (probs * values[targets.to(weights.device)]).sum(dim=1)
        return values[0] + weights.sum() * 0


def compile_world(
    world: OrderWorld, horizon: int, max_nodes: int = 100_000,
    max_seconds: float | None = None, prune_impossible: bool = False,
) -> CompiledWorld:
    if horizon < 0 or max_nodes < 1:
        raise ValueError("horizon must be nonnegative and max_nodes must be positive")
    if max_seconds is not None and (not isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    started = perf_counter()

    def check_deadline(nodes):
        elapsed = perf_counter() - started
        if max_seconds is not None and elapsed > max_seconds:
            raise GraphLimitError("max_seconds", nodes, elapsed)

    bounds = {}

    def can_finish(state, remaining):
        if not prune_impossible:
            return True
        if state not in bounds:
            bounds[state] = completion_lower_bound(world, state)
        bound = bounds[state]
        return bound is not None and bound <= remaining

    initial = world.initial_state()
    feature_count = observation(world, initial, horizon, horizon).size
    layers = [(initial,) if can_finish(initial, horizon) else ()]
    transitions = []
    features = []
    node_count = len(layers[0])
    pruned_edges = 0
    for elapsed in range(horizon):
        current = layers[-1]
        next_ids = {}
        next_states = []
        targets = np.empty((len(current), world.action_count), dtype=np.int64)
        for state_id, state in enumerate(current):
            check_deadline(node_count + len(next_states))
            for action in range(world.action_count):
                following = world.step(state, action)
                if not can_finish(following, horizon - elapsed - 1):
                    targets[state_id, action] = -1
                    pruned_edges += 1
                    continue
                if following not in next_ids:
                    if node_count + len(next_states) + 1 > max_nodes:
                        raise GraphLimitError(
                            "max_nodes", node_count + len(next_states), perf_counter() - started,
                        )
                    next_ids[following] = len(next_states)
                    next_states.append(following)
                targets[state_id, action] = next_ids[following]
        features.append(torch.tensor(np.asarray([
            observation(world, state, horizon - elapsed, horizon) for state in current
        ], dtype=np.float64).reshape(len(current), feature_count)))
        transitions.append(torch.from_numpy(targets))
        layers.append(tuple(next_states))
        node_count += len(next_states)

    check_deadline(node_count)
    rewards = torch.tensor([world.reward(state) for state in layers[-1]], dtype=torch.float64)
    return CompiledWorld(
        tuple(layers), tuple(transitions), tuple(features), rewards,
        pruned_edges, len(bounds),
    )
