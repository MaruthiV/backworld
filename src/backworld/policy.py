import numpy as np
import torch

from backworld.environment import ACTION_COUNT, OrderState, OrderWorld


FEATURE_COUNT = 16


def observation(
    world: OrderWorld, state: OrderState, remaining: int, horizon: int,
) -> np.ndarray:
    capacity = max(1, world.initial_stock + world.replenishments)
    return np.concatenate((
        np.eye(4, dtype=np.float64)[list(state.stages)].reshape(-1),
        np.asarray(state.stock, dtype=np.float64) / capacity,
        [state.replenishments / max(1, world.replenishments)],
        [remaining / max(1, horizon)],
        world.target_orders,
        [world.shared_inventory, 1.0],
    ))


def initial_weights(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.15, (ACTION_COUNT, FEATURE_COUNT))


def probabilities(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    logits = weights @ features
    values = np.exp(logits - logits.max())
    return values / values.sum()


def action_score(features: np.ndarray, probs: np.ndarray, action: int) -> np.ndarray:
    score = -probs[:, None] * features[None, :]
    score[action] += features
    return score


class SoftmaxPolicy(torch.nn.Module):
    def __init__(self, weights: np.ndarray):
        super().__init__()
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (ACTION_COUNT, FEATURE_COUNT) or not np.isfinite(weights).all():
            raise ValueError("weights must be a finite action-by-feature matrix")
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(features @ self.weights.T, dim=-1)
