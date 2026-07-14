# Project Status

## Current Phase

Phase 1 dataset-generation milestone implementation.

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
- Unit tests have been added for deterministic reset, movement, collisions, scripted motion, and local observation boundaries.
- Unit tests now also cover dataset determinism, alignment, manifest output, and save/load behavior.
- README has been updated with local setup, visualization run instructions, controls, and dataset-generation commands.
- Git ignore rules now keep generated `.npz` datasets out of version control while preserving repository documentation in `docs/`.

## Current Task

Keeping generated dataset artifacts out of Git while preserving version-controlled dataset documentation.

## Next Task

Begin the next narrow Phase 1 milestone: world-model training scaffolding that consumes the generated observation-action-next-observation datasets without introducing planning yet.

## Known Issues

- Local Python version is `3.14`, which may create package compatibility issues for some ecosystem packages as the project grows.
- The current milestone uses placeholder packages for future modules that are not implemented yet.
- `pygame` availability on Python `3.14` remains a potential installation risk on some machines even though the code path is now in place.
- The current dataset uses a simple seeded-random observer policy and does not yet cover richer behavior policies.
- No model training or planning logic exists yet.
- The generated manifest and dataset card in `data/generated/` are reproducible outputs and should not be treated as the primary version-controlled documentation source.

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
- Preserve a `reset()` and `step()` interface that can later support data generation and planning without a redesign.

## Handoff Notes

- Read `docs/architecture.md` before modifying code.
- Stay within the Phase 1 scope defined in `docs/roadmap.md`.
- Update this file after meaningful completed work so the current repo state remains easy to hand off.
