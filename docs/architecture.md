# Architecture

## Goals

The architecture should be minimal, explainable, and extensible enough to support:

- a 2D partially observed navigation environment
- offline trajectory collection
- supervised world-model training
- local CPU-friendly experimentation
- later model-based planning without rewriting the core simulation

The initial target is a clean Phase 1 baseline, not a research platform.

## System Overview

The project is organized around five core layers:

1. `Environment`
   - Simulates the grid-world state, obstacles, scripted moving agents, and agent movement.
2. `Observation`
   - Extracts the observer agent's local `7x7` view and converts it into model-ready tensors.
3. `Dataset`
   - Stores transition sequences generated from environment rollouts for supervised training.
4. `World Model`
   - Learns to predict the next local observation from recent observation history and action input.
5. `Visualization and Evaluation`
   - Renders the environment and model behavior for debugging, demos, and qualitative review.

Phase 2 planning should call into the trained world model and environment interfaces without changing their core schemas.

## Core Schemas

### Environment Schema

The full environment state should remain separate from the agent's partial observation.

Recommended conceptual schema:

```text
EnvironmentConfig
- grid_height: int
- grid_width: int
- local_view_size: int = 7
- max_episode_steps: int
- obstacle_density: float
- num_scripted_agents: int
- seed: int | None

EnvironmentState
- step_index: int
- observer_position: (int, int)
- observer_heading: str | None
- static_obstacles: binary grid [H, W]
- scripted_agents: list of agent states
- goal_position: (int, int) | None
- done: bool
- outcome: str | None
```

Notes:

- `observer_heading` is optional in Phase 1. Add it only if directional observation becomes necessary.
- `goal_position` may be omitted at first if the earliest milestone focuses on prediction rather than task completion.

### Observation Schema

The agent sees a local window centered on its own position.

Recommended representation:

```text
Observation
- grid: int tensor [C, 7, 7]
- visibility_mask: binary tensor [1, 7, 7]   # optional if padding/unknown cells must be explicit
- metadata:
  - agent_position_global: (int, int)        # optional for logging only
  - step_index: int
```

Recommended channel design for `grid`:

- channel 0: static obstacles
- channel 1: scripted moving agents
- channel 2: observer self-location
- channel 3: goal or target marker if used
- optional unknown/out-of-bounds channel if explicit boundary encoding helps training

This multi-channel encoding is preferred over token IDs because it stays simple for PyTorch CNN-style encoders and is easy to inspect visually.

### Action Schema

Keep the first action space discrete and minimal:

```text
ActionSpace
- 0: stay
- 1: up
- 2: down
- 3: left
- 4: right
```

Action tensors for training should use either:

- integer class IDs for storage
- one-hot vectors for model input

### Model Schema

The Phase 1 world model predicts the next local observation:

```text
WorldModelInput
- observation_t: float tensor [B, C, 7, 7]
- action_t: float tensor [B, A]
- recurrent_state: float tensor [B, H] or GRU hidden state

WorldModelOutput
- predicted_observation_t_plus_1: float tensor [B, C, 7, 7]
- recurrent_state_next
```

For sequence training:

```text
SequenceBatch
- observations: float tensor [B, T, C, 7, 7]
- actions: int or one-hot tensor [B, T, A] or [B, T]
- targets: float tensor [B, T, C, 7, 7]
- episode_id: optional metadata
```

## Minimal Model Architecture

Phase 1 model:

1. Convolutional or lightweight MLP encoder over the `7x7` observation.
2. Action embedding or one-hot projection.
3. Concatenation of encoded observation and encoded action.
4. GRU core over time.
5. Decoder projecting latent state back to observation channels.

Why this is the right baseline:

- Small enough for CPU debugging.
- Natural fit for partially observed sequential prediction.
- Easy to port to Colab GPU later.
- Simple to extend with latent rollouts or reward heads in Phase 2.

## Module Responsibilities and Interfaces

Recommended modules and responsibilities:

### `environment`

- owns grid generation
- owns obstacle placement
- owns scripted moving-agent logic
- applies observer actions
- exposes `reset()` and `step(action)`

Expected interface:

```text
reset(seed=None) -> initial_observation, info
step(action) -> next_observation, reward_or_placeholder, done, info
get_full_state() -> EnvironmentState
render_state() -> render-friendly snapshot
```

Even if reward is unused in Phase 1, returning a placeholder value keeps the interface compatible with Phase 2.

### `observation`

- converts full state into a local observer-centric view
- handles out-of-bounds padding
- converts symbolic state into channel tensors

Expected interface:

```text
extract_local_observation(full_state, config) -> Observation
encode_observation(observation) -> tensor
```

### `data`

- rolls out episodes
- saves trajectories
- builds train/validation splits
- yields sequence batches

Expected interface:

```text
generate_episode(...) -> transition list
save_dataset(path, episodes, metadata)
load_dataset(path) -> dataset object
build_sequence_dataset(dataset, sequence_length) -> training-ready dataset
```

### `models`

- defines encoder, recurrent core, decoder
- exposes training-time forward pass
- later exposes rollout inference for planning

Expected interface:

```text
forward(observations, actions, hidden_state=None) -> predictions, hidden_state
predict_next(observation, action, hidden_state=None) -> prediction, hidden_state
```

### `training`

- training loop
- validation loop
- checkpoint save/load
- metric logging

Expected interface:

```text
train_epoch(...)
validate_epoch(...)
save_checkpoint(...)
load_checkpoint(...)
```

### `visualization`

- Pygame rendering of world state
- optional side-by-side display of ground-truth and predicted local observations
- demo playback for saved episodes or trained models

Expected interface:

```text
run_simulation_viewer(...)
run_prediction_viewer(...)
```

### `planning` (Phase 2)

- uses the trained world model for short-horizon action evaluation
- compares candidate action sequences under simplified objectives

Expected interface:

```text
plan_action(current_observation, hidden_state, objective_state) -> action
```

## Dataset Structure

Keep the dataset simple and portable across local and Colab environments.

Recommended episode-level contents:

```text
EpisodeRecord
- episode_id: str
- config_id: str
- seed: int
- observations: [T, C, 7, 7]
- actions: [T]
- next_observations: [T, C, 7, 7]
- dones: [T]
- optional full-state summaries for debugging only
```

Recommended dataset metadata:

```text
DatasetMetadata
- dataset_version: str
- local_view_size: int
- action_space: list[str]
- observation_channels: list[str]
- generation_notes: str
- train_split
- val_split
```

Storage guidance:

- Use NumPy-backed arrays or PyTorch-friendly serialized tensors for trajectory data.
- Use `pandas` only for experiment summaries, not as the main trajectory format.
- Avoid prematurely introducing heavy dataset tooling.

## Folder Structure

Do not create this full structure yet, but use it as the intended target:

```text
project-root/
- README.md
- AGENTS.md
- CLAUDE.md
- docs/
  - architecture.md
  - roadmap.md
  - technical-decisions.md
  - project-status.md
- src/
  - environment/
  - observation/
  - data/
  - models/
  - training/
  - visualization/
  - planning/
- tests/
- scripts/
- data/
  - raw/
  - processed/
- artifacts/
  - checkpoints/
  - figures/
```

Reasoning:

- `src/` isolates implementation code.
- `tests/` supports confidence as the project evolves.
- `data/` separates generated datasets from code.
- `artifacts/` separates checkpoints and outputs from reusable source files.

## Local Development Workflow

Local Windows development should focus on fast iteration and correctness:

1. Implement and debug the environment on CPU.
2. Validate local observation extraction with deterministic test cases.
3. Generate a small local dataset.
4. Train tiny smoke-test runs locally on CPU.
5. Use Pygame to visually verify environment behavior and model predictions.

Local goals:

- short runs
- deterministic seeds
- debuggable outputs
- no assumption of GPU access

## Colab Training Workflow

Heavy training should be deferred to Google Colab once the local pipeline is stable.

Recommended workflow:

1. Generate or package datasets locally in a stable on-disk format.
2. Upload dataset artifacts and training notebook or script to Colab.
3. Install the same minimal dependency set in Colab.
4. Train larger sequence models using GPU acceleration.
5. Export checkpoints back to the repository's artifact conventions.
6. Bring trained checkpoints back to local Windows for visualization and planning demos.

Important principle:

The same model code and dataset schema should work both locally and in Colab. Only execution scale should change.

## Non-Goals for Initial Architecture

The initial architecture should not include:

- distributed training
- online RL infrastructure
- vector databases
- web services
- Docker orchestration
- cloud deployment
- complex experiment tracking platforms

These would add overhead without improving the one-month delivery target.
