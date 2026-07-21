# Technical Decisions

## Decision Style

This project prioritizes decisions that are:

- realistic within one month
- easy to explain in a portfolio walkthrough
- lightweight enough for local CPU development
- compatible with later Google Colab acceleration

## Core Technology Choices

### Python for implementation

Decision:

- Use Python as the primary implementation language.

Reason:

- Best ecosystem fit for PyTorch, NumPy, Pygame, pandas, matplotlib, and scikit-learn.
- Easiest path for sharing code between local development and Colab.

### PyTorch for the world model

Decision:

- Use PyTorch for model definition, sequence training, checkpointing, and inference.

Reason:

- Standard and well-supported for recurrent sequence models.
- Easy CPU debugging locally and straightforward GPU acceleration in Colab.
- Minimal friction for experimenting with encoder-decoder architectures.

### NumPy for environment-state and dataset primitives

Decision:

- Use NumPy for grid manipulation, deterministic rollout data, and lightweight serialization support.

Reason:

- Efficient and simple for grid-world logic.
- Reduces unnecessary tensor overhead in non-training code.

### Pygame for visualization

Decision:

- Use Pygame for the interactive local visualization layer.

Reason:

- Well-suited to 2D grid-based simulations.
- Fast to iterate on locally.
- Good fit for a professional portfolio demo without adding web stack complexity.
- Keeps visualization separate from model training concerns.

### GRU for the recurrent world model core

Decision:

- Use an encoder-GRU-decoder architecture as the baseline world model.

Reason:

- GRUs are simpler and lighter than LSTMs while still strong enough for short sequential dependencies.
- Better aligned with limited local hardware and one-month scope than transformer-based sequence models.
- Natural baseline for partial observability where temporal memory matters.

### Google Colab for heavier training

Decision:

- Use local Windows development for implementation and debugging, then use Google Colab GPU for heavier training runs.

Reason:

- Local machine has no dedicated GPU and only `16 GB` RAM.
- Colab is sufficient for a portfolio-scale sequence model without introducing cloud infrastructure overhead.
- Keeps the project inexpensive and accessible.

## Python Version Compatibility Strategy

### Local version: Python 3.14

Decision:

- Do not automatically change the local environment.
- Treat Python `3.14` as the current development context, but document a fallback path if key ML libraries lag in support.

Reason:

- Some packages, especially PyTorch and Pygame, may not immediately provide stable wheels for Python `3.14`.
- Silent environment changes would create unnecessary risk and confusion.

Strategy:

1. Try to keep repository code version-agnostic across modern Python `3.x`.
2. Prefer dependency pins that are known to work in Colab and likely to work on common stable Python releases.
3. If local installation problems appear on Python `3.14`, recommend a fallback environment such as Python `3.11` or `3.12` for local development.
4. Keep this recommendation documented rather than enforced automatically.

Current compatibility risk:

- Python `3.14` is likely the highest technical risk in the current setup because binary wheel availability may lag for PyTorch and Pygame.

## Project Structure Philosophy

Decision:

- Use a modular but minimal structure separating environment, data, models, training, visualization, planning, tests, and artifacts.

Reason:

- Keeps responsibilities clear.
- Supports future planning work without forcing a large refactor.
- Prevents the codebase from collapsing into a single experimental script.

## Data Strategy

Decision:

- Start with offline trajectory collection and supervised next-observation prediction.

Reason:

- Easier to validate than end-to-end reinforcement learning.
- Produces reusable datasets for debugging and training.
- Makes the value of the world model measurable before decision-making is introduced.

## Evaluation Strategy

Decision:

- Combine quantitative prediction metrics with qualitative visualization.

Reason:

- Loss alone may hide failure modes in spatial prediction tasks.
- A visual demo is especially important for a portfolio project.

## Sparse Positive-Class Handling

Decision:

- Use capped channel-aware weighted BCE and optional balanced sequence sampling for training.

Reason:

- Overall binary accuracy is dominated by empty cells in this dataset and can hide poor positive-cell prediction.
- Scripted-agent cells and some obstacle cells are sparse enough that plain BCE can under-train them.
- Weighted BCE directly increases the penalty for missed positive cells while keeping the existing model architecture unchanged.
- A cap on `pos_weight` keeps the loss numerically stable and prevents one rare channel from dominating optimization.
- Optional balanced sampling increases exposure to sequences containing scripted agents without changing validation or test distributions.

Constraint:

- Compute prevalence and class weights using only the training split.
- Keep validation and test sampling unchanged so comparisons remain honest.

## Explicitly Rejected or Postponed Technologies

### Docker

Status:

- Rejected for initial scope.

Reason:

- Adds setup overhead without materially improving a one-month local-plus-Colab workflow.

### Databases

Status:

- Rejected for initial scope.

Reason:

- Trajectory datasets can be stored as files. A database would be unnecessary complexity.

### Web frameworks and browser visualization

Status:

- Postponed.

Reason:

- Pygame is faster to build and sufficient for the initial demo.

### Cloud deployment platforms

Status:

- Rejected for current scope.

Reason:

- The project goal is research-style prototyping and a local demo, not production deployment.

### Transformer-based world model as baseline

Status:

- Postponed.

Reason:

- Higher implementation and tuning cost than justified for the first month.
- A GRU baseline is simpler, faster, and easier to explain.

### Full reinforcement learning first

Status:

- Explicitly rejected as the initial approach.

Reason:

- Skipping directly to RL would make debugging harder and weaken the portfolio narrative.
- The planning baseline should prove that the learned model has practical value before RL is attempted.

### Complex experiment tracking stacks

Status:

- Postponed.

Reason:

- Lightweight local logging and saved artifacts are enough for the current scope.

## Decision Summary

The project intentionally favors:

- simple over clever
- modular over monolithic
- supervised world-model validation before RL
- local CPU debugging plus Colab GPU training
- portfolio clarity over research sprawl
