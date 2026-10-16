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


def initial_weights(seed: int = 7, world: OrderWorld | None = None) -> np.ndarray:
    world = OrderWorld() if world is None else world
    shape = (world.action_count, 6 * world.order_count + 4)
    return np.random.default_rng(seed).normal(0, 0.15, shape)


def validate_weights(weights: np.ndarray, world: OrderWorld | None = None):
    if weights.ndim != 2 or not np.isfinite(weights).all():
        raise ValueError("weights must be a finite action-by-feature matrix")
    if world is None:
        orders, remainder = divmod(weights.shape[0] - 2, 4)
        valid = orders >= 1 and remainder == 0 and weights.shape[1] == 6 * orders + 4
    else:
        valid = weights.shape == (world.action_count, 6 * world.order_count + 4)
    if not valid:
        raise ValueError("weight dimensions do not match the order world")


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
        validate_weights(weights)
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(features @ self.weights.T, dim=-1)
