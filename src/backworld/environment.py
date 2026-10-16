from dataclasses import dataclass
from operator import index


ACTION_NAMES = (
    "reserve_0", "pay_0", "ship_0", "cancel_0",
    "reserve_1", "pay_1", "ship_1", "cancel_1",
    "replenish", "wait",
)
ACTION_COUNT = len(ACTION_NAMES)


@dataclass(frozen=True, slots=True)
class OrderState:
    stages: tuple[int, ...]
    stock: tuple[int, ...]
    replenishments: int


@dataclass(frozen=True, slots=True)
class OrderWorld:
    shared_inventory: bool = True
    initial_stock: int = 1
    replenishments: int = 1
    target_orders: tuple[bool, ...] = (True, True)

    def __post_init__(self):
        if self.initial_stock < 0 or self.replenishments < 0:
            raise ValueError("stock and replenishments must be nonnegative")
        if not self.target_orders:
            raise ValueError("at least one order is required")

    @property
    def order_count(self) -> int:
        return len(self.target_orders)

    @property
    def action_count(self) -> int:
        return 4 * self.order_count + 2

    def initial_state(self) -> OrderState:
        stock = (self.initial_stock,) + (
            (0 if self.shared_inventory else self.initial_stock),
        ) * (self.order_count - 1)
        return OrderState((0,) * self.order_count, stock, self.replenishments)

    def step(self, state: OrderState, action: int) -> OrderState:
        action = index(action)
        if not 0 <= action < self.action_count:
            raise ValueError("action is outside the supported action set")
        if action == self.action_count - 1:
            return state
        if action == self.action_count - 2:
            if state.replenishments == 0:
                return state
            return OrderState(
                state.stages,
                (state.stock[0] + 1,) + state.stock[1:],
                state.replenishments - 1,
            )

        order, operation = divmod(action, 4)
        pool = 0 if self.shared_inventory else order
        stage = state.stages[order]
        stages = list(state.stages)
        stock = list(state.stock)
        if operation == 0 and stage == 0 and stock[pool] > 0:
            stages[order] = 1
            stock[pool] -= 1
        elif operation == 1 and stage == 1:
            stages[order] = 2
        elif operation == 2 and stage == 2:
            stages[order] = 3
        elif operation == 3 and stage in (1, 2):
            stages[order] = 0
            stock[pool] += 1
        else:
            return state
        return OrderState(tuple(stages), tuple(stock), state.replenishments)

    def reward(self, state: OrderState) -> float:
        expected = tuple(3 if target else 0 for target in self.target_orders)
        return float(state.stages == expected)
