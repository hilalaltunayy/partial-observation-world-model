# Dataset

## Overview

This document describes the current generated Phase 1 rollout dataset for the `Partial Observation World Model for Autonomous Navigation` project.

The dataset is produced from the existing deterministic `20x20` grid-world environment and is intended for supervised world-model training. Each saved transition contains the observer's current local observation, the chosen observer action, and the next local observation after the environment step.

## Actual Generated Dataset

The current generated dataset was created with:

```powershell
python -m src.data.generate --episodes 300 --steps 50 --seed 42
```

This produces:

- `data/generated/train.npz`
- `data/generated/validation.npz`
- `data/generated/test.npz`
- `data/generated/manifest.json`
- `data/generated/DATASET_CARD.md`

Only the documentation in this `docs/` directory is intended to be version-controlled. The generated `.npz` split files remain ignored by Git.

## Environment

- Grid size: `20x20`
- Local observation size: `7x7`
- Static obstacles: deterministic per episode seed
- Observer agent: 1
- Scripted moving agents: 2
- Scripted-agent behavior: deterministic fixed action cycles
- Observer policy during data generation: seeded random action sampling

## Dataset Schema

Each transition stores the following arrays:

- `observations`: current local observation, shape `[N, 4, 7, 7]`, dtype `int8`
- `actions`: observer action id, shape `[N]`, dtype `int8`
- `next_observations`: next local observation, shape `[N, 4, 7, 7]`, dtype `int8`
- `episode_ids`: episode id string, shape `[N]`, dtype `<U32`
- `time_steps`: time step within the episode, shape `[N]`, dtype `int16`
- `episode_seeds`: deterministic episode seed, shape `[N]`, dtype `int32`
- `observer_positions`: observer `(row, col)`, shape `[N, 2]`, dtype `int16`
- `scripted_agent_positions`: scripted-agent positions, shape `[N, 2, 2]`, dtype `int16`

## Action Encoding

- `0`: `up`
- `1`: `down`
- `2`: `left`
- `3`: `right`

## Observation Channels

- Channel `0`: static obstacles
- Channel `1`: scripted agents
- Channel `2`: observer location
- Channel `3`: out-of-bounds or unknown cells

This preserves the existing environment observation format used elsewhere in the repository.

## Split Sizes

The current generated dataset uses `300` total episodes and `50` steps per episode:

- Train: `210` episodes, `10500` transitions
- Validation: `45` episodes, `2250` transitions
- Test: `45` episodes, `2250` transitions

Per-split array shapes:

- Train:
  - `observations`: `[10500, 4, 7, 7]`
  - `actions`: `[10500]`
  - `next_observations`: `[10500, 4, 7, 7]`
  - `observer_positions`: `[10500, 2]`
  - `scripted_agent_positions`: `[10500, 2, 2]`
- Validation:
  - `observations`: `[2250, 4, 7, 7]`
  - `actions`: `[2250]`
  - `next_observations`: `[2250, 4, 7, 7]`
  - `observer_positions`: `[2250, 2]`
  - `scripted_agent_positions`: `[2250, 2, 2]`
- Test:
  - `observations`: `[2250, 4, 7, 7]`
  - `actions`: `[2250]`
  - `next_observations`: `[2250, 4, 7, 7]`
  - `observer_positions`: `[2250, 2]`
  - `scripted_agent_positions`: `[2250, 2, 2]`

## Seed Allocation

The current generated dataset uses disjoint deterministic episode seeds:

- Train seeds: `42` through `251`
- Validation seeds: `252` through `296`
- Test seeds: `297` through `341`

This guarantees that train, validation, and test do not share episode seeds.

## Regeneration

To regenerate the current dataset:

```powershell
python -m src.data.generate --episodes 300 --steps 50 --seed 42
```

To generate a different dataset while keeping the same schema:

```powershell
python -m src.data.generate --episodes 300 --steps 50 --seed 100
```

## Limitations

- The observer policy is a simple seeded-random action sampler, not a learned or task-directed policy.
- The dataset contains only observation-action-next-observation transitions.
- Rewards, goals, planning traces, and model predictions are not part of this Phase 1 dataset.
- The action space currently does not include a `stay` action.
- The dataset is useful for next-observation prediction, but it does not yet emphasize strategically meaningful trajectories.

## Intended Use

This dataset is the Phase 1 training input for the future world model. The next milestone can consume:

- `observations`
- `actions`
- `next_observations`

to train an encoder-GRU-decoder baseline that predicts the next local observation from the current local observation and action.
