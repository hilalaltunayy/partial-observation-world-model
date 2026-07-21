# Partial Observation World Model

This repository contains a staged portfolio project for a partial-observation autonomous navigation system. The current implementation includes the deterministic environment baseline, a professional dark-themed Pygame viewer, deterministic rollout dataset generation, a local CPU-compatible world-model training scaffold, and checkpoint-based Phase 1 evaluation utilities.

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
python -m src.visualization.app --checkpoint checkpoints/colab_world_model/world_model_best.pt --metadata checkpoints/colab_world_model/world_model_best.metadata.json --device cpu
```

The visualization now keeps the full ground-truth map visible while also showing:

- the current local observation
- the model-predicted next observation
- the real next observation after the action
- a compact prediction-error view
- a model panel with checkpoint name, one-step prediction loss, and selected device

If the checkpoint files are missing or incompatible, the viewer stays open and shows a clear model-status message instead of crashing.

Controls:

- `Space`: pause or resume playback
- `R`: reset the episode using the current seed
- `Arrow keys`: manually move the observer and advance scripted agents
- `N`: advance one step while paused
- `Esc`: exit

## Run Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

## Generate the Dataset

```powershell
.venv\Scripts\python.exe -m src.data.generate --episodes 300 --steps 50 --seed 42
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

## World-Model Smoke Test

```powershell
.venv\Scripts\python.exe -m src.training.train_world_model --data-dir data/generated --device cpu --smoke-test
```

This runs a tiny deterministic one-epoch CPU pass over a small subset of the generated train and validation sequences and saves the best validation checkpoint under `checkpoints/`.

## World-Model Training

```powershell
.venv\Scripts\python.exe -m src.training.train_world_model --data-dir data/generated --epochs 2 --batch-size 32 --sequence-length 8 --device cpu
```

The training scaffold:

- loads train and validation NPZ splits
- builds fixed-length contiguous sequences per episode
- trains a small encoder-GRU-decoder model on CPU
- reports total loss, overall binary accuracy, and per-channel binary accuracies
- saves the best validation checkpoint and metadata under `checkpoints/`

## Class-Imbalance Correction

The current dataset is highly sparse for positive cells in some channels. Before this milestone, the model could achieve high overall binary accuracy while still performing poorly on sparse positives:

- scripted-agent F1 could collapse to `0`
- obstacle recall could remain very low
- the persistence baseline could outperform the trained model on scripted-agent F1

The updated training pipeline now supports:

- training-split-only channel prevalence statistics
- channel-aware weighted BCE with capped positive-class weights
- optional deterministic balanced sampling for scripted-agent-positive sequences
- checkpoint selection using validation mean F1 for static obstacles and scripted agents, with validation loss as a tie-breaker

### Local Weighted Smoke Test

```powershell
.venv\Scripts\python.exe -m src.training.train_world_model --data-dir data/generated --device cpu --smoke-test --loss-mode weighted_bce --pos-weight-cap 25 --use-balanced-sampling --agent-sequence-sampling-weight 4
```

### Recommended Colab Retraining Direction

Use the Colab notebook with:

- `loss_mode = "weighted_bce"`
- a positive-weight cap such as `25.0`
- `use_balanced_sampling = True`
- an agent-sequence sampling weight such as `4.0`

## World-Model Evaluation

```powershell
.venv\Scripts\python.exe -m src.evaluation.evaluate_world_model --checkpoint checkpoints/colab_world_model/world_model_best.pt --metadata checkpoints/colab_world_model/world_model_best.metadata.json --data-dir data/generated --split test --device cpu
```

The evaluation CLI:

- rebuilds the world model from the metadata JSON
- loads the trained weights on CPU or another selected device
- evaluates the chosen split with BCE loss and overall binary accuracy
- reports per-channel precision, recall, F1 score, and IoU
- compares the trained model against a persistence baseline that predicts the next observation as unchanged

## Google Colab Notebook

Use `notebooks/world_model_training_colab.ipynb` for a fresh Colab workflow that:

- clones the repository and checks out the `dev` branch
- installs the project and plotting dependency
- regenerates the deterministic dataset when needed
- trains the existing encoder-GRU-decoder world model with configurable settings
- evaluates the best checkpoint and exports it as a zip bundle

The notebook reuses the repository's existing dataset CLI, model code, training utilities, and checkpoint format instead of duplicating implementation inside notebook cells.

## Current Scope

Implemented now:

- deterministic `20x20` grid world
- static obstacle generation
- observer movement with collision handling
- two scripted moving agents
- configurable local observation extraction
- dark-themed Pygame viewer with full-grid and local-view panels
- deterministic rollout dataset generation with manifest and dataset card
- CPU-compatible Phase 1 world-model scaffold and smoke-training CLI
- class-imbalance-aware weighted training and optional balanced sequence sampling
- reusable checkpoint loading and evaluation CLI
- checkpoint-backed Pygame prediction viewer
- unit tests for core environment behavior

Not implemented yet:

- planning or reinforcement learning
- anomaly detection
