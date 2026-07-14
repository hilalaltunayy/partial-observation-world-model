from __future__ import annotations

import numpy as np


def extract_local_observation(state: object, config: object) -> np.ndarray:
    view_size = int(getattr(config, "local_view_size"))
    half_window = view_size // 2
    observer_row, observer_col = getattr(state, "observer_position")
    obstacle_grid = getattr(state, "static_obstacles")
    scripted_agents = getattr(state, "scripted_agents")

    observation = np.zeros((4, view_size, view_size), dtype=np.int8)

    for local_row in range(view_size):
        for local_col in range(view_size):
            global_row = observer_row + local_row - half_window
            global_col = observer_col + local_col - half_window

            if not (0 <= global_row < obstacle_grid.shape[0] and 0 <= global_col < obstacle_grid.shape[1]):
                observation[3, local_row, local_col] = 1
                continue

            observation[0, local_row, local_col] = obstacle_grid[global_row, global_col]
            if any(agent.position == (global_row, global_col) for agent in scripted_agents):
                observation[1, local_row, local_col] = 1
            if (global_row, global_col) == (observer_row, observer_col):
                observation[2, local_row, local_col] = 1

    return observation
