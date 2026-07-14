from __future__ import annotations

import argparse
import os
from typing import Final

import pygame

try:
    from environment import Action, GridWorldConfig, GridWorldEnvironment
    from observation import extract_local_observation
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.visualization.app`
    from src.environment import Action, GridWorldConfig, GridWorldEnvironment
    from src.observation import extract_local_observation

from .viewer import GridWorldViewer


FPS: Final[int] = 60
AUTO_STEP_INTERVAL_MS: Final[int] = 220
# The environment has no "stay" action yet, so autoplay repeats the last action
# and falls back to a deterministic default before the first manual move.
DEFAULT_AUTO_ACTION: Final[Action] = Action.RIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grid-world Pygame visualization.")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed used for deterministic resets.")
    parser.add_argument("--fps", type=int, default=FPS, help="Display refresh rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GridWorldConfig(seed=args.seed)
    env = GridWorldEnvironment(config)

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    try:
        viewer = GridWorldViewer(config)
        window = pygame.display.set_mode(viewer.window_size)
        pygame.display.set_caption("Partial Observation World Model - Grid World Viewer")
        clock = pygame.time.Clock()

        observation, info = env.reset(seed=args.seed)
        episode_seed = int(info["seed"])
        last_action: Action | None = None
        paused = True
        running = True
        last_auto_step_ms = pygame.time.get_ticks()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        observation, info = env.reset(seed=episode_seed)
                        episode_seed = int(info["seed"])
                        last_action = None
                        paused = True
                        last_auto_step_ms = pygame.time.get_ticks()
                    elif event.key == pygame.K_n and paused:
                        step_action = last_action or DEFAULT_AUTO_ACTION
                        observation, last_action = _advance_environment(env, step_action)
                        last_auto_step_ms = pygame.time.get_ticks()
                    else:
                        mapped_action = _map_key_to_action(event.key)
                        if mapped_action is not None:
                            observation, last_action = _advance_environment(env, mapped_action)
                            last_auto_step_ms = pygame.time.get_ticks()

            now_ms = pygame.time.get_ticks()
            if not paused and now_ms - last_auto_step_ms >= AUTO_STEP_INTERVAL_MS:
                step_action = last_action or DEFAULT_AUTO_ACTION
                observation, last_action = _advance_environment(env, step_action)
                last_auto_step_ms = now_ms

            state = env.get_full_state()
            current_observation = extract_local_observation(state, env.config)
            viewer.draw(window, state, current_observation, last_action, episode_seed, paused)
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        pygame.quit()


def _advance_environment(env: GridWorldEnvironment, action: Action) -> tuple[object, Action]:
    observation, _, done, _ = env.step(action)
    if done:
        return observation, action
    return observation, action


def _map_key_to_action(key: int) -> Action | None:
    if key == pygame.K_UP:
        return Action.UP
    if key == pygame.K_DOWN:
        return Action.DOWN
    if key == pygame.K_LEFT:
        return Action.LEFT
    if key == pygame.K_RIGHT:
        return Action.RIGHT
    return None


if __name__ == "__main__":
    main()
