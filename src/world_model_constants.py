CHANNEL_NAMES: tuple[str, ...] = (
    "static_obstacles",
    "scripted_agents",
    "observer_location",
    "out_of_bounds",
)

CHANNEL_DISPLAY_NAMES: dict[str, str] = {
    "static_obstacles": "static obstacles",
    "scripted_agents": "scripted agents",
    "observer_location": "observer",
    "out_of_bounds": "out-of-bounds / unknown",
}

STATIC_OBSTACLES_CHANNEL_INDEX: int = 0
SCRIPTED_AGENTS_CHANNEL_INDEX: int = 1
OBSERVER_CHANNEL_INDEX: int = 2
OUT_OF_BOUNDS_CHANNEL_INDEX: int = 3
