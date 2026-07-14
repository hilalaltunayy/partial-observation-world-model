# Partial Observation World Model

This repository contains a staged portfolio project for a partial-observation autonomous navigation system. The current implementation includes the deterministic environment baseline and a professional dark-themed Pygame viewer for inspecting the full `20x20` world alongside the observer's local view.

## Local Setup

1. Create and activate a virtual environment.
2. Install the project in editable mode with development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If Python `3.14` fails to install compatible wheels for required packages, especially `pygame`, use Python `3.11` or `3.12` locally instead. The code is intentionally kept compatible with that fallback workflow.

## Run the Visualization

```powershell
python -m src.visualization.app
```

Optional:

```powershell
python -m src.visualization.app --seed 7
```

Controls:

- `Space`: pause or resume playback
- `R`: reset the episode using the current seed
- `Arrow keys`: manually move the observer and advance scripted agents
- `N`: advance one step while paused
- `Esc`: exit

## Run Tests

```powershell
python -m pytest
```

## Generate the Dataset

```powershell
python -m src.data.generate --episodes 300 --steps 50 --seed 42
```

This command generates deterministic rollout splits and writes:

- `data/generated/train.npz`
- `data/generated/validation.npz`
- `data/generated/test.npz`
- `data/generated/manifest.json`
- `data/generated/DATASET_CARD.md`

The generated transitions contain:

- current local observation
- observer action
- next local observation
- episode id
- time step
- episode seed
- observer position
- scripted-agent positions

## Current Scope

Implemented now:

- deterministic `20x20` grid world
- static obstacle generation
- observer movement with collision handling
- two scripted moving agents
- configurable local observation extraction
- dark-themed Pygame viewer with full-grid and local-view panels
- deterministic rollout dataset generation with manifest and dataset card
- unit tests for core environment behavior

Not implemented yet:

- PyTorch models
- dataset generation
- training
- planning or reinforcement learning
- anomaly detection
