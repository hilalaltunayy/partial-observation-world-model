# AGENTS.md

## Purpose

This repository is a one-month portfolio project for a `Partial Observation World Model for Autonomous Navigation`. The goal is to deliver a realistic, well-documented baseline before adding advanced research features.

## Working Rules

1. Read the project documentation before modifying code:
   - `docs/architecture.md`
   - `docs/roadmap.md`
   - `docs/technical-decisions.md`
   - `docs/project-status.md`
2. Prefer small, reviewable changes over large refactors.
3. Do not add implementation beyond the current phase defined in `docs/roadmap.md`.
4. Add or update tests for important behavior whenever code is introduced or changed.
5. Keep dependencies minimal and compatible with both local Windows development and later Google Colab training.
6. If Python `3.14` causes package compatibility problems, document the fallback recommendation instead of changing the local environment automatically.
7. Do not overengineer. Choose the simplest design that preserves a clean upgrade path.
8. Update `docs/project-status.md` after meaningful completed work so the next contributor has an accurate handoff.
9. Never commit or push automatically. Git commits and pushes must always be user-driven.

## Implementation Expectations

- Treat Phase 1 as the primary delivery target:
  - 2D grid-world simulation
  - configurable `7x7` local observation
  - static obstacles and scripted moving agents
  - encoder-GRU-decoder world model predicting the next local observation
  - professional Pygame visualization
- Do not introduce reinforcement learning until the planning baseline exists and works.
- Treat anomaly detection as a future stretch goal, not part of the first working baseline.

## Quality Bar

- Keep modules narrow in responsibility.
- Prefer explicit data schemas and typed interfaces where practical.
- Keep training and visualization decoupled.
- Ensure code can run locally on CPU with modest memory usage.
- Design data artifacts so they can be generated locally and trained later in Colab without format changes.
