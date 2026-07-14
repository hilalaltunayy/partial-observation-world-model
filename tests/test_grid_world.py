import numpy as np

from environment import Action, GridWorldConfig, GridWorldEnvironment
from observation import extract_local_observation


def build_env(seed: int = 7, obstacle_count: int = 0, observer_start_position=(10, 10)) -> GridWorldEnvironment:
    config = GridWorldConfig(
        seed=seed,
        obstacle_count=obstacle_count,
        observer_start_position=observer_start_position,
    )
    env = GridWorldEnvironment(config)
    env.reset()
    return env


def test_reset_is_deterministic_for_same_seed() -> None:
    env_a = GridWorldEnvironment(GridWorldConfig(seed=11, obstacle_count=10))
    env_b = GridWorldEnvironment(GridWorldConfig(seed=11, obstacle_count=10))

    obs_a, info_a = env_a.reset()
    obs_b, info_b = env_b.reset()

    state_a = env_a.get_full_state()
    state_b = env_b.get_full_state()

    assert info_a["seed"] == info_b["seed"] == 11
    assert np.array_equal(obs_a, obs_b)
    assert state_a.observer_position == state_b.observer_position
    assert [agent.position for agent in state_a.scripted_agents] == [agent.position for agent in state_b.scripted_agents]
    assert np.array_equal(state_a.static_obstacles, state_b.static_obstacles)


def test_observer_moves_in_open_space() -> None:
    env = build_env(obstacle_count=0, observer_start_position=(10, 10))

    env.step(Action.UP)
    env.step(Action.LEFT)

    state = env.get_full_state()
    assert state.observer_position == (9, 9)


def test_observer_cannot_move_outside_map() -> None:
    env = build_env(obstacle_count=0, observer_start_position=(0, 0))

    env.step(Action.UP)
    env.step(Action.LEFT)

    state = env.get_full_state()
    assert state.observer_position == (0, 0)


def test_observer_cannot_move_through_obstacles() -> None:
    env = build_env(obstacle_count=0, observer_start_position=(10, 10))
    env._state.static_obstacles[9, 10] = 1

    env.step(Action.UP)

    updated_state = env.get_full_state()
    assert updated_state.observer_position == (10, 10)


def test_scripted_agents_move_deterministically() -> None:
    env = build_env(obstacle_count=0)

    env.step(Action.RIGHT)

    state = env.get_full_state()
    positions = [agent.position for agent in state.scripted_agents]
    assert positions == [(3, 4), (16, 15)]


def test_local_observation_marks_out_of_bounds_without_leaking_state() -> None:
    env = build_env(obstacle_count=0, observer_start_position=(0, 0))
    state = env.get_full_state()
    state.static_obstacles[19, 19] = 1

    observation = extract_local_observation(state, env.config)

    assert observation.shape == (4, 7, 7)
    assert observation[2, 3, 3] == 1
    assert observation[3, 0, 0] == 1
    assert observation[0, 0, 0] == 0
    assert observation[1, 0, 0] == 0
    assert observation[2, 0, 0] == 0
    assert observation[0].sum() == 0


def test_local_observation_includes_visible_obstacles_only() -> None:
    env = build_env(obstacle_count=0, observer_start_position=(10, 10))
    env._state.static_obstacles[9, 10] = 1
    env._state.static_obstacles[0, 0] = 1

    observation = env.step(Action.RIGHT)[0]

    assert observation[0, 2, 2] == 1
    assert observation[3].sum() == 0
    assert observation[0].sum() == 1
