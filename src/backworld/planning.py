from collections import deque
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from backworld.environment import OrderWorld


@dataclass(frozen=True)
class PlanResult:
    status: str
    actions: tuple[int, ...] | None
    discovered_states: int
    expanded_states: int
    seconds: float


def shortest_plan(
    world: OrderWorld, horizon: int, max_nodes: int = 50_000,
    max_seconds: float = 10.0,
) -> PlanResult:
    if horizon < 0 or max_nodes < 1 or not isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("invalid planning bounds")
    started = perf_counter()
    initial = world.initial_state()
    parents = {initial: None}
    pending = deque([(initial, 0)])
    expanded = 0
    while pending:
        if perf_counter() - started > max_seconds:
            return PlanResult("time_limit", None, len(parents), expanded, perf_counter() - started)
        state, depth = pending.popleft()
        if world.reward(state):
            actions = []
            while parents[state] is not None:
                state, action = parents[state]
                actions.append(action)
            return PlanResult(
                "solved", tuple(reversed(actions)), len(parents), expanded, perf_counter() - started,
            )
        if depth == horizon:
            continue
        expanded += 1
        for action in range(world.action_count):
            following = world.step(state, action)
            if following in parents:
                continue
            if len(parents) >= max_nodes:
                return PlanResult("node_limit", None, len(parents), expanded, perf_counter() - started)
            parents[following] = (state, action)
            pending.append((following, depth + 1))
    return PlanResult(
        "unreachable_within_horizon", None, len(parents), expanded, perf_counter() - started,
    )
