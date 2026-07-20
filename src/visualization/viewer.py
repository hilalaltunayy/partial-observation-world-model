from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

try:
    from environment import Action, GridWorldConfig, GridWorldState
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.visualization.app`
    from src.environment import Action, GridWorldConfig, GridWorldState

try:
    from visualization.prediction import PredictionState
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.visualization.app`
    from src.visualization.prediction import PredictionState


@dataclass(frozen=True)
class ViewerTheme:
    background: tuple[int, int, int] = (14, 18, 24)
    panel_background: tuple[int, int, int] = (22, 28, 36)
    panel_border: tuple[int, int, int] = (58, 71, 88)
    grid_line: tuple[int, int, int] = (44, 53, 66)
    text_primary: tuple[int, int, int] = (230, 236, 244)
    text_secondary: tuple[int, int, int] = (150, 162, 179)
    empty_cell: tuple[int, int, int] = (30, 36, 45)
    obstacle: tuple[int, int, int] = (86, 99, 117)
    observer: tuple[int, int, int] = (74, 211, 255)
    scripted_agent: tuple[int, int, int] = (255, 167, 87)
    unknown: tuple[int, int, int] = (61, 49, 76)
    accent: tuple[int, int, int] = (125, 211, 252)


class GridWorldViewer:
    def __init__(self, config: GridWorldConfig) -> None:
        self.config = config
        self.theme = ViewerTheme()
        self.margin = 24
        self.section_gap = 24
        self.grid_cell_size = 28
        self.local_cell_size = 48
        self.info_panel_width = 280
        self.side_panel_gap = 18
        self.footer_height = 52
        self.preview_gap = 10
        self.preview_cell_size = 14

        self._title_font = pygame.font.SysFont("segoeui", 22, bold=True)
        self._label_font = pygame.font.SysFont("segoeui", 17, bold=True)
        self._body_font = pygame.font.SysFont("consolas", 16)
        self._small_font = pygame.font.SysFont("segoeui", 14)
        self.title_top_padding = 10
        self.title_bottom_gap = 14
        self.footer_text_offset = 8

        full_grid_pixels = self.config.grid_width * self.grid_cell_size
        local_grid_pixels = self.config.local_view_size * self.local_cell_size
        self.content_height = max(full_grid_pixels, local_grid_pixels)
        self.title_band_height = self.title_top_padding + self._title_font.get_height() + self.title_bottom_gap
        self.info_panel_height = 228
        self.model_panel_height = 126
        self.legend_panel_height = self.content_height - self.info_panel_height - self.model_panel_height - (2 * self.side_panel_gap)
        content_width = (
            full_grid_pixels
            + self.section_gap
            + local_grid_pixels
            + self.section_gap
            + self.info_panel_width
        )

        self.window_size = (
            self.margin * 2 + content_width,
            self.margin * 2 + self.title_band_height + self.content_height + self.footer_height,
        )

    def draw(
        self,
        surface: pygame.Surface,
        state: GridWorldState,
        local_observation: np.ndarray,
        last_action: Action | None,
        episode_seed: int,
        is_paused: bool,
        prediction_state: PredictionState,
    ) -> None:
        surface.fill(self.theme.background)

        title_y = self.margin + self.title_top_padding
        content_top = self.margin + self.title_band_height
        full_grid_origin = (self.margin, content_top)
        local_grid_origin = (
            full_grid_origin[0] + self.config.grid_width * self.grid_cell_size + self.section_gap,
            content_top,
        )
        info_origin = (
            local_grid_origin[0] + self.config.local_view_size * self.local_cell_size + self.section_gap,
            content_top,
        )

        self._draw_panel_title(surface, "Ground-Truth Grid World", (full_grid_origin[0], title_y))
        self._draw_panel_title(
            surface,
            f"Observer {self.config.local_view_size}x{self.config.local_view_size} Local View",
            (local_grid_origin[0], title_y),
        )
        self._draw_panel_title(surface, "Episode Info", (info_origin[0], title_y))

        self._draw_full_grid(surface, state, full_grid_origin)
        self._draw_local_view(surface, local_observation, local_grid_origin)
        self._draw_prediction_strip(surface, prediction_state, local_grid_origin)
        self._draw_info_panel(surface, state, last_action, episode_seed, is_paused, info_origin, prediction_state)
        self._draw_footer(surface)

    def _draw_panel_title(self, surface: pygame.Surface, text: str, position: tuple[int, int]) -> None:
        title = self._title_font.render(text, True, self.theme.text_primary)
        surface.blit(title, position)

    def _draw_panel_box(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        radius: int = 12,
    ) -> None:
        pygame.draw.rect(surface, self.theme.panel_background, rect, border_radius=radius)
        pygame.draw.rect(surface, self.theme.panel_border, rect, width=1, border_radius=radius)

    def _draw_full_grid(
        self,
        surface: pygame.Surface,
        state: GridWorldState,
        origin: tuple[int, int],
    ) -> None:
        panel_rect = pygame.Rect(
            origin[0] - 10,
            origin[1] - 10,
            self.config.grid_width * self.grid_cell_size + 20,
            self.config.grid_height * self.grid_cell_size + 20,
        )
        self._draw_panel_box(surface, panel_rect)

        scripted_positions = {agent.position for agent in state.scripted_agents}
        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                cell_rect = pygame.Rect(
                    origin[0] + col * self.grid_cell_size,
                    origin[1] + row * self.grid_cell_size,
                    self.grid_cell_size,
                    self.grid_cell_size,
                )
                color = self.theme.empty_cell
                if state.static_obstacles[row, col] == 1:
                    color = self.theme.obstacle
                pygame.draw.rect(surface, color, cell_rect, border_radius=4)
                pygame.draw.rect(surface, self.theme.grid_line, cell_rect, width=1, border_radius=4)

                if (row, col) in scripted_positions:
                    self._draw_scripted_agent(surface, cell_rect)
                if (row, col) == state.observer_position:
                    self._draw_observer(surface, cell_rect)

    def _draw_local_view(
        self,
        surface: pygame.Surface,
        local_observation: np.ndarray,
        origin: tuple[int, int],
    ) -> None:
        panel_rect = pygame.Rect(
            origin[0] - 10,
            origin[1] - 10,
            self.config.local_view_size * self.local_cell_size + 20,
            self.config.local_view_size * self.local_cell_size + 20,
        )
        self._draw_panel_box(surface, panel_rect)

        for row in range(self.config.local_view_size):
            for col in range(self.config.local_view_size):
                cell_rect = pygame.Rect(
                    origin[0] + col * self.local_cell_size,
                    origin[1] + row * self.local_cell_size,
                    self.local_cell_size,
                    self.local_cell_size,
                )
                is_unknown = local_observation[3, row, col] == 1
                color = self.theme.unknown if is_unknown else self.theme.empty_cell
                if local_observation[0, row, col] == 1:
                    color = self.theme.obstacle
                pygame.draw.rect(surface, color, cell_rect, border_radius=6)
                pygame.draw.rect(surface, self.theme.grid_line, cell_rect, width=1, border_radius=6)

                if is_unknown:
                    self._draw_unknown_marker(surface, cell_rect)
                    continue
                if local_observation[1, row, col] == 1:
                    self._draw_scripted_agent(surface, cell_rect)
                if local_observation[2, row, col] == 1:
                    self._draw_observer(surface, cell_rect)

    def _draw_prediction_strip(
        self,
        surface: pygame.Surface,
        prediction_state: PredictionState,
        origin: tuple[int, int],
    ) -> None:
        strip_top = origin[1] + self.config.local_view_size * self.local_cell_size + 18
        strip_label = self._label_font.render("One-Step Prediction", True, self.theme.text_primary)
        surface.blit(strip_label, (origin[0], strip_top))

        grid_pixels = self.config.local_view_size * self.preview_cell_size
        preview_top = strip_top + self._label_font.get_height() + 10
        preview_specs = [
            ("Predicted next", prediction_state.predicted_next_observation, False),
            ("Real next", prediction_state.real_next_observation, False),
            ("Error", prediction_state.absolute_prediction_error, True),
        ]

        current_x = origin[0]
        for label, observation, is_error in preview_specs:
            panel_rect = pygame.Rect(current_x, preview_top, grid_pixels + 16, grid_pixels + 42)
            self._draw_panel_box(surface, panel_rect, radius=10)
            label_surface = self._small_font.render(label, True, self.theme.text_primary)
            surface.blit(label_surface, (panel_rect.left + 8, panel_rect.top + 8))
            if observation is None:
                placeholder = self._small_font.render("Unavailable", True, self.theme.text_secondary)
                surface.blit(
                    placeholder,
                    (
                        panel_rect.left + 8,
                        panel_rect.top + 16 + max(0, (grid_pixels - placeholder.get_height()) // 2),
                    ),
                )
            else:
                self._draw_compact_grid(
                    surface,
                    observation,
                    (panel_rect.left + 8, panel_rect.top + 24),
                    self.preview_cell_size,
                    is_error=is_error,
                )
            current_x += panel_rect.width + self.preview_gap

    def _draw_scripted_agent(self, surface: pygame.Surface, cell_rect: pygame.Rect) -> None:
        radius = max(6, cell_rect.width // 4)
        center = cell_rect.center
        pygame.draw.circle(surface, self.theme.scripted_agent, center, radius)
        pygame.draw.circle(surface, self.theme.panel_background, center, max(2, radius // 2))

    def _draw_observer(self, surface: pygame.Surface, cell_rect: pygame.Rect) -> None:
        center_x, center_y = cell_rect.center
        half = max(7, cell_rect.width // 4)
        points = [
            (center_x, center_y - half),
            (center_x + half, center_y),
            (center_x, center_y + half),
            (center_x - half, center_y),
        ]
        pygame.draw.polygon(surface, self.theme.observer, points)

    def _draw_unknown_marker(self, surface: pygame.Surface, cell_rect: pygame.Rect) -> None:
        inset = 8
        start_a = (cell_rect.left + inset, cell_rect.top + inset)
        end_a = (cell_rect.right - inset, cell_rect.bottom - inset)
        start_b = (cell_rect.right - inset, cell_rect.top + inset)
        end_b = (cell_rect.left + inset, cell_rect.bottom - inset)
        pygame.draw.line(surface, self.theme.text_secondary, start_a, end_a, width=2)
        pygame.draw.line(surface, self.theme.text_secondary, start_b, end_b, width=2)

    def _draw_info_panel(
        self,
        surface: pygame.Surface,
        state: GridWorldState,
        last_action: Action | None,
        episode_seed: int,
        is_paused: bool,
        origin: tuple[int, int],
        prediction_state: PredictionState,
    ) -> None:
        panel_rect = pygame.Rect(origin[0], origin[1], self.info_panel_width, self.info_panel_height)
        self._draw_panel_box(surface, panel_rect)

        status_value = "Paused" if is_paused else "Running"
        last_action_value = self._action_label(last_action)
        lines = [
            ("Current step", str(state.step_index)),
            ("Observer position", f"{state.observer_position}"),
            ("Last action", last_action_value),
            ("Episode seed", str(episode_seed)),
            ("Playback state", status_value),
        ]

        top_padding = 18
        bottom_padding = 12
        row_height = (panel_rect.height - top_padding - bottom_padding) // len(lines)
        value_right = panel_rect.right - 16
        for index, (label, value) in enumerate(lines):
            y = panel_rect.top + top_padding + index * row_height
            label_surface = self._label_font.render(label, True, self.theme.text_secondary)
            value_surface = self._body_font.render(value, True, self.theme.text_primary)
            label_y = y + max(0, (row_height - label_surface.get_height()) // 2)
            value_x = value_right - value_surface.get_width()
            value_y = y + max(0, (row_height - value_surface.get_height()) // 2)
            surface.blit(label_surface, (panel_rect.left + 16, label_y))
            surface.blit(value_surface, (value_x, value_y))

            if index < len(lines) - 1:
                divider_y = y + row_height - 4
                pygame.draw.line(
                    surface,
                    self.theme.grid_line,
                    (panel_rect.left + 16, divider_y),
                    (panel_rect.right - 16, divider_y),
                    width=1,
                )

        model_top = panel_rect.bottom + self.side_panel_gap
        model_rect = pygame.Rect(origin[0], model_top, self.info_panel_width, self.model_panel_height)
        self._draw_panel_box(surface, model_rect)
        model_title = self._label_font.render("Model Panel", True, self.theme.text_primary)
        surface.blit(model_title, (model_rect.left + 16, model_rect.top + 14))

        loss_text = "N/A" if prediction_state.prediction_loss is None else f"{prediction_state.prediction_loss:.4f}"
        model_lines = [
            ("Checkpoint", prediction_state.checkpoint_name),
            ("Device", prediction_state.device_label),
            ("Prediction loss", loss_text),
        ]
        if prediction_state.status_message is not None:
            model_lines.append(("Status", "Checkpoint unavailable"))

        text_y = model_rect.top + 46
        for label, value in model_lines:
            label_surface = self._small_font.render(label, True, self.theme.text_secondary)
            value_surface = self._small_font.render(value, True, self.theme.text_primary)
            surface.blit(label_surface, (model_rect.left + 16, text_y))
            surface.blit(value_surface, (model_rect.left + 132, text_y))
            text_y += 24

        if prediction_state.status_message is not None:
            status_message = self._truncate_text(prediction_state.status_message, 32)
            status_surface = self._small_font.render(status_message, True, self.theme.scripted_agent)
            surface.blit(status_surface, (model_rect.left + 16, model_rect.bottom - 24))

        legend_top = model_rect.bottom + self.side_panel_gap
        legend_rect = pygame.Rect(origin[0], legend_top, self.info_panel_width, self.legend_panel_height)
        self._draw_panel_box(surface, legend_rect)
        legend_title = self._label_font.render("Legend", True, self.theme.text_primary)
        surface.blit(legend_title, (legend_rect.left + 16, legend_rect.top + 14))

        legend_rows = [
            ("Observer", self.theme.observer),
            ("Scripted agent", self.theme.scripted_agent),
            ("Obstacle", self.theme.obstacle),
            ("Empty cell", self.theme.empty_cell),
            ("Unknown / out-of-bounds", self.theme.unknown),
        ]
        y = legend_rect.top + 56
        row_gap = 34
        for text, color in legend_rows:
            swatch = pygame.Rect(legend_rect.left + 16, y, 20, 20)
            pygame.draw.rect(surface, color, swatch, border_radius=5)
            pygame.draw.rect(surface, self.theme.grid_line, swatch, width=1, border_radius=5)
            line = self._small_font.render(text, True, self.theme.text_primary)
            line_y = y + (swatch.height - line.get_height()) // 2
            surface.blit(line, (swatch.right + 12, line_y))
            y += row_gap

    def _draw_compact_grid(
        self,
        surface: pygame.Surface,
        observation: np.ndarray,
        origin: tuple[int, int],
        cell_size: int,
        *,
        is_error: bool,
    ) -> None:
        for row in range(self.config.local_view_size):
            for col in range(self.config.local_view_size):
                cell_rect = pygame.Rect(
                    origin[0] + col * cell_size,
                    origin[1] + row * cell_size,
                    cell_size,
                    cell_size,
                )
                if is_error:
                    color = self.theme.accent if observation[row, col] == 1 else self.theme.empty_cell
                    pygame.draw.rect(surface, color, cell_rect, border_radius=3)
                    pygame.draw.rect(surface, self.theme.grid_line, cell_rect, width=1, border_radius=3)
                    continue

                is_unknown = observation[3, row, col] == 1
                color = self.theme.unknown if is_unknown else self.theme.empty_cell
                if observation[0, row, col] == 1:
                    color = self.theme.obstacle
                pygame.draw.rect(surface, color, cell_rect, border_radius=3)
                pygame.draw.rect(surface, self.theme.grid_line, cell_rect, width=1, border_radius=3)
                if not is_unknown and observation[1, row, col] == 1:
                    self._draw_scripted_agent(surface, cell_rect)
                if not is_unknown and observation[2, row, col] == 1:
                    self._draw_observer(surface, cell_rect)

    def _draw_footer(self, surface: pygame.Surface) -> None:
        footer_text = "Controls: Space pause/resume  |  R reset  |  Arrow keys move  |  N single-step while paused  |  Esc exit"
        footer_surface = self._small_font.render(footer_text, True, self.theme.text_secondary)
        footer_y = self.window_size[1] - self.footer_height + self.footer_text_offset
        surface.blit(footer_surface, (self.margin, footer_y))

    def _action_label(self, action: Action | None) -> str:
        if action is None:
            return "None"
        return action.name.title()

    def _truncate_text(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return f"{text[: max_length - 3]}..."
