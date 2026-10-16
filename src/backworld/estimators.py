import numpy as np

from backworld.environment import OrderState, OrderWorld
from backworld.policy import action_score, observation, probabilities, validate_weights


class ExactSuffix:
    def __init__(
        self, world: OrderWorld, weights: np.ndarray, horizon: int,
        exact_steps: int, baseline: float = 0.0,
    ):
        if not 0 <= exact_steps <= horizon:
            raise ValueError("exact_steps must lie between zero and the horizon")
        weights = np.array(weights, dtype=np.float64, copy=True)
        validate_weights(weights, world)
        if not np.isfinite(baseline):
            raise ValueError("baseline must be finite")
        weights.flags.writeable = False
        self.world = world
        self.weights = weights
        self.horizon = horizon
        self.exact_steps = exact_steps
        self.baseline = baseline
        self.policy_cache = {}
        self.suffix_cache = {}
        self.policy_evaluations = 0
        self.transitions = 0

    def _policy(self, state: OrderState, remaining: int):
        key = (state, remaining)
        if key not in self.policy_cache:
            features = observation(self.world, state, remaining, self.horizon)
            self.policy_cache[key] = (features, probabilities(self.weights, features))
            self.policy_evaluations += 1
        return self.policy_cache[key]

    def continuation(self, state: OrderState, remaining: int):
        if not 0 <= remaining <= self.horizon:
            raise ValueError("remaining steps are outside this estimator's horizon")
        key = (state, remaining)
        if key in self.suffix_cache:
            return self.suffix_cache[key]
        if remaining == 0:
            result = (self.world.reward(state), np.zeros_like(self.weights))
        else:
            features, probs = self._policy(state, remaining)
            value = 0.0
            derivative = np.zeros_like(self.weights)
            for action in range(self.world.action_count):
                following = self.world.step(state, action)
                self.transitions += 1
                child_value, child_derivative = self.continuation(following, remaining - 1)
                value += probs[action] * child_value
                derivative += probs[action] * (
                    child_derivative + child_value * action_score(features, probs, action)
                )
            result = (value, derivative)
        result[1].flags.writeable = False
        self.suffix_cache[key] = result
        return result

    def estimate(self, rng: np.random.Generator):
        state = self.world.initial_state()
        score = np.zeros_like(self.weights)
        for remaining in range(self.horizon, self.exact_steps, -1):
            features, probs = self._policy(state, remaining)
            action = int(rng.choice(self.world.action_count, p=probs))
            score += action_score(features, probs, action)
            state = self.world.step(state, action)
            self.transitions += 1
        value, derivative = self.continuation(state, self.exact_steps)
        return value, (value - self.baseline) * score + derivative
