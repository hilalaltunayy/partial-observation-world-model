# CLAUDE.md

## Repository Context

This repository hosts a one-month portfolio project named `Partial Observation World Model for Autonomous Navigation`.

The intended delivery order is:
1. Build a solid Phase 1 baseline.
2. Add model-based planning using the learned world model.
3. Consider a small reinforcement learning extension only after the planning baseline works.

## Read Before Changing Code

Use these documents as the source of truth:

- Architecture: `docs/architecture.md`
- Delivery plan and milestones: `docs/roadmap.md`
- Technical choices and tradeoffs: `docs/technical-decisions.md`
- Current progress and handoff state: `docs/project-status.md`

Read the documentation before making changes so implementation stays aligned with the intended scope.

## Working Instructions

1. Prefer small, reviewable changes.
2. Do not add implementation beyond the current roadmap phase unless explicitly requested.
3. Add tests for important behavior when creating or changing code.
4. Keep dependencies minimal and compatible with local Windows development and Google Colab.
5. If Python `3.14` causes package support issues, document a fallback recommendation instead of changing the environment automatically.
6. Avoid overengineering. Keep the project realistically deliverable within one month.
7. Update `docs/project-status.md` after meaningful completed work.
8. Never commit or push automatically.

## Project Boundaries

- No Docker unless explicitly requested later.
- No database layer.
- No web framework.
- No cloud deployment setup.
- No reinforcement learning work before the planning baseline is functioning.

## Collaboration Notes

- Architecture and scope decisions should be recorded in `docs/technical-decisions.md`.
- If implementation reality diverges from the roadmap, update the roadmap and project status together.
- Favor maintainable CPU-friendly baselines over ambitious but brittle designs.
