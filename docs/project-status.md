# Project Status

## Current Phase

Phase 1 completion milestone with checkpoint evaluation and prediction-backed visualization.

## Completed Work

- Repository setup is complete.
- Initial architecture, roadmap, and technical decision documents are in place.
- Minimal `src/` package structure has been created for:
  - `environment`
  - `observation`
  - `data`
  - `models`
  - `training`
  - `visualization`
  - `planning`
- First environment milestone has been implemented:
  - deterministic `20x20` grid world
  - static obstacles
  - one observer agent
  - two scripted moving agents
  - four-direction movement with boundary and obstacle collision handling
  - configurable `7x7` local observation with explicit out-of-bounds masking
- Pygame visualization milestone has been implemented:
  - dark-themed desktop interface
  - full ground-truth grid and local `7x7` view shown side by side
  - compact info panel with step, position, last action, seed, and playback state
  - keyboard controls for pause, reset, manual movement, single-step advance, and exit
  - rendering kept separate from environment logic in `src/visualization`
- Dataset-generation milestone has been implemented:
  - deterministic rollout collection in `src/data`
  - compressed NumPy train, validation, and test split export
  - machine-readable `manifest.json`
  - human-readable `DATASET_CARD.md`
  - CLI entry point for reproducible split generation
  - version-controlled dataset reference document in `docs/dataset.md`
- World-model scaffold milestone has been implemented:
  - PyTorch sequence dataset loader for grouped per-episode fixed-length sequences
  - configurable encoder-GRU-decoder world model
  - BCE-with-logits training loss for binary observation channels
  - overall and per-channel binary accuracy reporting
  - CPU-compatible training CLI with smoke-test mode
  - best-validation checkpoint and metadata saving
  - local smoke-training verified under `.venv\Scripts\python.exe`
- Google Colab training notebook milestone has been implemented:
  - fresh-runtime repository clone and `dev` branch checkout flow
  - automatic CUDA-or-CPU device selection with runtime summary output
  - deterministic dataset presence check with CLI regeneration fallback
  - configurable training cell that reuses the existing model and training utilities
  - best-checkpoint reload, validation/test evaluation, and compact qualitative plots
  - checkpoint listing, zip export, and Colab download flow
- Phase 1 completion milestone has been implemented:
  - reusable checkpoint loader that reconstructs the world model from metadata JSON
  - test-split evaluation CLI with BCE loss, overall accuracy, and per-channel precision/recall/F1/IoU
  - persistence baseline comparison using the same metrics
  - Pygame integration for current-view, predicted-next, real-next, and prediction-error inspection
  - compact model-status panel with checkpoint name, device, and one-step prediction loss
- Unit tests have been added for deterministic reset, movement, collisions, scripted motion, and local observation boundaries.
- Unit tests now also cover dataset determinism, alignment, manifest output, save/load behavior, sequence loading, model forward shape, optimizer step, checkpoint reload, evaluation metrics, persistence baseline behavior, and prediction-state preparation.
- README has been updated with local setup, visualization run instructions, dataset-generation commands, world-model training commands, and checkpoint evaluation commands.
- Git ignore rules now keep generated `.npz` datasets out of version control while preserving repository documentation in `docs/`.

## Current Task

Stabilizing the completed Phase 1 baseline after adding checkpoint evaluation and local qualitative prediction inspection.

## Next Task

Begin the Phase 2A planning baseline using the now-verified trained checkpoint, evaluation CLI, and prediction-enabled viewer.

## Known Issues

- Local Python version is `3.14`, which may create package compatibility issues for some ecosystem packages as the project grows.
- The current milestone uses placeholder packages for future modules that are not implemented yet.
- `pygame` availability on Python `3.14` remains a potential installation risk on some machines even though the code path is now in place.
- The current dataset uses a simple seeded-random observer policy and does not yet cover richer behavior policies.
- The world-model scaffold is intentionally small and CPU-friendly, so prediction quality should be treated as a baseline rather than a finished model.
- No planning logic exists yet.
- The generated manifest and dataset card in `data/generated/` are reproducible outputs and should not be treated as the primary version-controlled documentation source.
- The current prediction viewer performs one-step local prediction inspection and is not yet a full policy or planning interface.

## Recent Decisions

- Keep the first implementation milestone focused strictly on deterministic environment mechanics and observation extraction.
- Represent local observations as four channels:
  - obstacles
  - scripted agents
  - observer location
  - out-of-bounds mask
- Add `pygame` only at the point where the visualization milestone is implemented.
- Use compressed `.npz` files plus a JSON manifest and dataset card for the first offline dataset format.
- Keep stable project-facing dataset documentation under `docs/`, while generated dataset artifacts remain reproducible outputs.
- Use a small MLP encoder plus GRU plus MLP decoder for the first local training baseline rather than heavier architectures.
- Preserve a `reset()` and `step()` interface that can later support data generation and planning without a redesign.

## Handoff Notes

- Read `docs/architecture.md` before modifying code.
- Stay within the Phase 1 scope defined in `docs/roadmap.md`.
- Update this file after meaningful completed work so the current repo state remains easy to hand off.
