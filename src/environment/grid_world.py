from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import random

import numpy as np

from observation.local_view import extract_local_observation


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class ScriptedAgentSpec:
    start_position: tuple[int, int]
    action_cycle: tuple[Action, ...]


@dataclass
class ScriptedAgentState:
    agent_id: str
    position: tuple[int, int]
    action_cycle: tuple[Action, ...]
    cycle_index: int = 0


@dataclass(frozen=True)
class GridWorldConfig:
    grid_height: int = 20
    grid_width: int = 20
    local_view_size: int = 7
    max_episode_steps: int = 100
    obstacle_count: int = 24
    num_scripted_agents: int = 2
    seed: int = 0
    observer_start_position: tuple[int, int] = (10, 10)
    scripted_agent_specs: tuple[ScriptedAgentSpec, ...] = (
        ScriptedAgentSpec(start_position=(3, 3), action_cycle=(Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)),
        ScriptedAgentSpec(start_position=(16, 16), action_cycle=(Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN)),
    )

    def __post_init__(self) -> None:
        if self.grid_height != 20 or self.grid_width != 20:
            raise ValueError("This milestone implements a fixed 20x20 grid world.")
        if self.local_view_size <= 0 or self.local_view_size % 2 == 0:
            raise ValueError("local_view_size must be a positive odd integer.")
        if self.num_scripted_agents != 2:
            raise ValueError("This milestone requires exactly two scripted moving agents.")
        if len(self.scripted_agent_specs) != self.num_scripted_agents:
            raise ValueError("scripted_agent_specs must match num_scripted_agents.")


@dataclass
class GridWorldState:
    step_index: int
    observer_position: tuple[int, int]
    static_obstacles: np.ndarray
    scripted_agents: list[ScriptedAgentState] = field(default_factory=list)
    done: bool = False
    outcome: str | None = None


class GridWorldEnvironment:
    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self._base_seed = self.config.seed
        self._random = random.Random(self._base_seed)
        self._state: GridWorldState | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        active_seed = self._base_seed if seed is None else seed
        self._random = random.Random(active_seed)
        static_obstacles = self._generate_obstacles()
        scripted_agents = [
            ScriptedAgentState(
                agent_id=f"scripted_{index}",
                position=spec.start_position,
                action_cycle=spec.action_cycle,
            )
            for index, spec in enumerate(self.config.scripted_agent_specs)
        ]

        self._state = GridWorldState(
            step_index=0,
            observer_position=self.config.observer_start_position,
            static_obstacles=static_obstacles,
            scripted_agents=scripted_agents,
            done=False,
            outcome=None,
        )
        observation = extract_local_observation(self._state, self.config)
        return observation, {"seed": active_seed}

    def step(self, action: Action | int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if self._state.done:
            observation = extract_local_observation(self._state, self.config)
            return observation, 0.0, True, self._build_info()

        normalized_action = Action(action)

        occupied_before = {agent.position for agent in self._state.scripted_agents}
        self._state.observer_position = self._attempt_move(
            self._state.observer_position,
            normalized_action,
            blocked_positions=occupied_before,
        )

        occupied_positions = {self._state.observer_position}
        updated_agents: list[ScriptedAgentState] = []
        for agent in self._state.scripted_agents:
            planned_action = agent.action_cycle[agent.cycle_index % len(agent.action_cycle)]
            other_agents = {
                existing.position for existing in self._state.scripted_agents if existing.agent_id != agent.agent_id
            }
            blocked_positions = occupied_positions | {state.position for state in updated_agents} | other_agents
            new_position = self._attempt_move(agent.position, planned_action, blocked_positions)
            updated_agents.append(
                ScriptedAgentState(
                    agent_id=agent.agent_id,
                    position=new_position,
                    action_cycle=agent.action_cycle,
                    cycle_index=(agent.cycle_index + 1) % len(agent.action_cycle),
                )
            )
            occupied_positions.add(new_position)

        self._state.scripted_agents = updated_agents
        self._state.step_index += 1
        self._state.done = self._state.step_index >= self.config.max_episode_steps
        if self._state.done:
            self._state.outcome = "max_steps_reached"

        observation = extract_local_observation(self._state, self.config)
        return observation, 0.0, self._state.done, self._build_info()

    def get_full_state(self) -> GridWorldState:
        if self._state is None:
            raise RuntimeError("Environment must be reset before reading state.")
        return GridWorldState(
            step_index=self._state.step_index,
            observer_position=self._state.observer_position,
            static_obstacles=self._state.static_obstacles.copy(),
            scripted_agents=[
                ScriptedAgentState(
                    agent_id=agent.agent_id,
                    position=agent.position,
                    action_cycle=agent.action_cycle,
                    cycle_index=agent.cycle_index,
                )
                for agent in self._state.scripted_agents
            ],
            done=self._state.done,
            outcome=self._state.outcome,
        )

    def render_state(self) -> dict[str, object]:
        state = self.get_full_state()
        return {
            "step_index": state.step_index,
            "observer_position": state.observer_position,
            "scripted_agents": [agent.position for agent in state.scripted_agents],
            "static_obstacles": state.static_obstacles.copy(),
        }

    def _generate_obstacles(self) -> np.ndarray:
        obstacle_grid = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.int8)
        reserved_positions = {
            self.config.observer_start_position,
            *(spec.start_position for spec in self.config.scripted_agent_specs),
        }
        candidates = [
            (row, col)
            for row in range(self.config.grid_height)
            for col in range(self.config.grid_width)
            if (row, col) not in reserved_positions
        ]
        selected_positions = self._random.sample(candidates, k=self.config.obstacle_count)
        for row, col in selected_positions:
            obstacle_grid[row, col] = 1
        return obstacle_grid

    def _attempt_move(
        self,
        position: tuple[int, int],
        action: Action,
        blocked_positions: set[tuple[int, int]],
    ) -> tuple[int, int]:
        delta_row, delta_col = ACTION_DELTAS[action]
        next_row = position[0] + delta_row
        next_col = position[1] + delta_col
        next_position = (next_row, next_col)

        if not self._in_bounds(next_position):
            return position
        if self._state is not None and self._state.static_obstacles[next_row, next_col] == 1:
            return position
        if next_position in blocked_positions:
            return position
        return next_position

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        row, col = position
        return 0 <= row < self.config.grid_height and 0 <= col < self.config.grid_width

    def _build_info(self) -> dict[str, object]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before reading info.")
        return {
            "step_index": self._state.step_index,
            "observer_position": self._state.observer_position,
            "scripted_agent_positions": [agent.position for agent in self._state.scripted_agents],
        }
