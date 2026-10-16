import pytest

from backworld.environment import ACTION_COUNT, OrderState, OrderWorld


def play(world, actions):
    state = world.initial_state()
    for action in actions:
        state = world.step(state, action)
    return state


def test_shared_stock_requires_replenishment_for_the_second_order():
    world = OrderWorld()
    first = play(world, [0, 1, 2])
    assert first == OrderState((3, 0), (0, 0), 1)
    assert world.step(first, 4) == first
    complete = first
    for action in [8, 4, 5, 6]:
        complete = world.step(complete, action)
    assert complete == OrderState((3, 3), (0, 0), 0)
    assert world.reward(complete) == 1


@pytest.mark.parametrize("actions", [[0, 3], [0, 1, 3]])
def test_cancellation_releases_one_unit_once(actions):
    world = OrderWorld()
    restored = play(world, actions)
    assert restored == world.initial_state()
    assert world.step(restored, 3) == restored


def test_payment_and_shipping_require_their_preceding_stages():
    world = OrderWorld()
    initial = world.initial_state()
    assert world.step(initial, 1) == initial
    assert world.step(initial, 2) == initial
    reserved = world.step(initial, 0)
    assert world.step(reserved, 2) == reserved
    shipped = play(world, [0, 1, 2])
    assert world.step(shipped, 3) == shipped


def test_a_protected_order_can_invalidate_a_provisional_success():
    world = OrderWorld(initial_stock=2, target_orders=(True, False))
    success = play(world, [0, 1, 2])
    assert world.reward(success) == 1
    damaged = world.step(success, 4)
    assert world.reward(damaged) == 0
    assert world.reward(world.step(damaged, 7)) == 1
    assert success.stages == (3, 0)


@pytest.mark.parametrize("shared", [True, False])
def test_reachable_states_conserve_stock_and_committed_units(shared):
    world = OrderWorld(shared_inventory=shared)
    initial_units = world.initial_stock * (1 if shared else 2)
    frontier = {world.initial_state()}
    for _ in range(7):
        following = set()
        for state in frontier:
            assert min(state.stock) >= 0
            assert 0 <= state.replenishments <= world.replenishments
            used_replenishments = world.replenishments - state.replenishments
            committed = sum(stage > 0 for stage in state.stages)
            assert sum(state.stock) + committed == initial_units + used_replenishments
            following.update(world.step(state, action) for action in range(ACTION_COUNT))
        frontier = following


@pytest.mark.parametrize("action", [-1, ACTION_COUNT])
def test_unknown_actions_are_errors(action):
    world = OrderWorld()
    with pytest.raises(ValueError):
        world.step(world.initial_state(), action)
