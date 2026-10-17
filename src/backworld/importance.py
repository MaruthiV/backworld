from dataclasses import dataclass
from math import exp

import numpy as np

from backworld.environment import OrderState, OrderWorld
from backworld.feasibility import completion_lower_bound
from backworld.policy import action_score, observation, validate_weights


@dataclass(frozen=True)
class FeasibleProposal:
    features: np.ndarray
    policy_probabilities: np.ndarray
    actions: tuple[int, ...]
    probabilities: np.ndarray
    destinations: tuple[OrderState, ...]
    log_normalizer: float


class FeasibleImportance:
    def __init__(self, world: OrderWorld, weights: np.ndarray, horizon: int):
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        validate_weights(self.weights, world)
        self.weights.flags.writeable = False
        self.world = world
        self.horizon = horizon
        self.bounds = {}
        self.proposals = {}
        self.policy_evaluations = 0
        self.transitions = 0
        self.sampled_steps = 0
        self.last_log_weight = -np.inf

    def _can_finish(self, state: OrderState, remaining: int) -> bool:
        if state not in self.bounds:
            self.bounds[state] = completion_lower_bound(self.world, state)
        bound = self.bounds[state]
        return bound is not None and bound <= remaining

    def proposal(self, state: OrderState, remaining: int) -> FeasibleProposal | None:
        if not 1 <= remaining <= self.horizon:
            raise ValueError("proposal needs a positive remaining horizon")
        key = (state, remaining)
        if key in self.proposals:
            return self.proposals[key]
        actions = []
        destinations = []
        for action in range(self.world.action_count):
            following = self.world.step(state, action)
            self.transitions += 1
            if self._can_finish(following, remaining - 1):
                actions.append(action)
                destinations.append(following)
        if not actions:
            self.proposals[key] = None
            return None

        features = observation(self.world, state, remaining, self.horizon)
        logits = self.weights @ features
        if not np.isfinite(logits).all():
            raise FloatingPointError("nonfinite policy logits")
        centered = logits - logits.max()
        log_probs = centered - np.log(np.exp(centered).sum())
        if not np.isfinite(log_probs).all():
            raise FloatingPointError("nonfinite policy log probabilities")
        log_normalizer = float(np.logaddexp.reduce(log_probs[actions]))
        proposal_probs = np.exp(log_probs[actions] - log_normalizer)
        if np.any(proposal_probs == 0):
            raise FloatingPointError("proposal probability underflow would remove success support")
        probs = np.exp(log_probs)
        for values in (features, probs, proposal_probs):
            values.flags.writeable = False
        proposal = FeasibleProposal(
            features, probs, tuple(actions), proposal_probs,
            tuple(destinations), log_normalizer,
        )
        self.policy_evaluations += 1
        self.proposals[key] = proposal
        return proposal

    def estimate(self, rng: np.random.Generator):
        state = self.world.initial_state()
        score = np.zeros_like(self.weights)
        self.last_log_weight = -np.inf
        if not self._can_finish(state, self.horizon):
            return 0.0, score
        log_weight = 0.0
        for remaining in range(self.horizon, 0, -1):
            proposal = self.proposal(state, remaining)
            if proposal is None:
                return 0.0, np.zeros_like(score)
            choice = int(rng.choice(len(proposal.actions), p=proposal.probabilities))
            score += action_score(
                proposal.features, proposal.policy_probabilities, proposal.actions[choice],
            )
            log_weight += proposal.log_normalizer
            state = proposal.destinations[choice]
            self.sampled_steps += 1
        self.last_log_weight = log_weight
        if not self.world.reward(state):
            return 0.0, np.zeros_like(score)
        value = exp(log_weight)
        if value == 0:
            raise FloatingPointError("importance weight underflow; positive return is not representable")
        gradient = value * score
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite importance estimate")
        return value, gradient
