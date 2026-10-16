from backworld.environment import OrderState, OrderWorld


def completion_lower_bound(world: OrderWorld, state: OrderState) -> int | None:
    steps = 0
    new_targets = []
    returned_stock = 0
    for order, (target, stage) in enumerate(zip(world.target_orders, state.stages)):
        if target:
            steps += 3 - stage
            if stage == 0:
                new_targets.append(order)
        elif stage == 3:
            return None
        elif stage in (1, 2):
            steps += 1
            returned_stock += 1

    if world.shared_inventory:
        replenishments = max(0, len(new_targets) - state.stock[0] - returned_stock)
    else:
        if any(state.stock[order] == 0 for order in new_targets if order != 0):
            return None
        replenishments = int(0 in new_targets and state.stock[0] == 0)
    if replenishments > state.replenishments:
        return None
    return steps + replenishments
