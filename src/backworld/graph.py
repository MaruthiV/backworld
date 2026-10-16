from dataclasses import dataclass

import numpy as np
import torch

from backworld.environment import ACTION_COUNT, OrderState, OrderWorld
from backworld.policy import SoftmaxPolicy, observation


@dataclass(frozen=True)
class CompiledWorld:
    layers: tuple[tuple[OrderState, ...], ...]
    transitions: tuple[torch.Tensor, ...]
    features: tuple[torch.Tensor, ...]
    rewards: torch.Tensor

    @property
    def node_count(self) -> int:
        return sum(len(layer) for layer in self.layers)

    @property
    def edge_count(self) -> int:
        return sum(transition.numel() for transition in self.transitions)

    def expected_return(self, policy: SoftmaxPolicy) -> torch.Tensor:
        weights = policy.weights
        values = self.rewards.to(weights)
        for features, targets in zip(reversed(self.features), reversed(self.transitions)):
            probs = policy(features.to(weights))
            values = (probs * values[targets.to(weights.device)]).sum(dim=1)
        return values[0] + weights.sum() * 0


def compile_world(world: OrderWorld, horizon: int, max_nodes: int = 100_000) -> CompiledWorld:
    if horizon < 0 or max_nodes < 1:
        raise ValueError("horizon must be nonnegative and max_nodes must be positive")
    layers = [(world.initial_state(),)]
    transitions = []
    features = []
    node_count = 1
    for elapsed in range(horizon):
        current = layers[-1]
        next_ids = {}
        next_states = []
        targets = np.empty((len(current), ACTION_COUNT), dtype=np.int64)
        for state_id, state in enumerate(current):
            for action in range(ACTION_COUNT):
                following = world.step(state, action)
                if following not in next_ids:
                    if node_count + len(next_states) + 1 > max_nodes:
                        raise ValueError("reachable graph exceeds max_nodes")
                    next_ids[following] = len(next_states)
                    next_states.append(following)
                targets[state_id, action] = next_ids[following]
        features.append(torch.tensor(np.stack([
            observation(world, state, horizon - elapsed, horizon) for state in current
        ]), dtype=torch.float64))
        transitions.append(torch.from_numpy(targets))
        layers.append(tuple(next_states))
        node_count += len(next_states)

    rewards = torch.tensor([world.reward(state) for state in layers[-1]], dtype=torch.float64)
    return CompiledWorld(tuple(layers), tuple(transitions), tuple(features), rewards)
