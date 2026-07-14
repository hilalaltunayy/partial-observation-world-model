# Roadmap

## Delivery Strategy

This project is intentionally divided into a strong standalone baseline and a clearly bounded extension phase.

The main success criterion is not "use every ML technique." The main success criterion is delivering a polished, understandable portfolio project that demonstrates systems thinking, model design, and practical execution.

## Phase 1: World Model Baseline

Phase 1 must be independently deliverable without reinforcement learning.

### Objective

Build a complete pipeline from simulation to learned next-observation prediction with a professional visualization.

### Scope

- 2D grid-world simulation
- observer agent with configurable `7x7` local observation
- static obstacles
- scripted moving agents
- rollout data generation
- encoder-GRU-decoder world model
- training and validation pipeline
- Pygame visualization for simulation playback
- optional side-by-side predicted-vs-true local observation viewer

### Completion Criteria

Phase 1 is complete when all of the following are true:

1. The environment resets and steps deterministically under fixed seeds.
2. The observer receives correct local `7x7` observations including edge handling.
3. Scripted moving agents behave consistently and visibly in the simulation.
4. A dataset of trajectory transitions can be generated and reloaded.
5. The world model trains end-to-end and predicts the next local observation better than a trivial baseline.
6. The project includes a polished Pygame demo showing the environment and at least one model output or evaluation mode.
7. Basic tests cover critical environment and observation logic.

### Measurable Outputs

- reproducible example episodes
- train/validation loss curves
- saved model checkpoint
- short demo or screenshot-ready visualization

### Fallback Delivery Point

If time becomes tight, Phase 1 can still ship as a strong portfolio piece if it includes:

- stable environment simulation
- correct partial observation extraction
- dataset generation
- trained world model with basic quantitative evaluation
- polished visualization

This fallback remains valid even if prediction quality is modest, provided the system is coherent and honest about limitations.

## Phase 2A: Model-Based Planning Baseline

Phase 2A should use the learned world model for simplified planning.

### Objective

Demonstrate that the learned model is useful for action selection, not only passive prediction.

### Scope

- short-horizon candidate action rollout using the learned world model
- simplified planning objective such as:
  - avoiding collisions
  - maintaining safe distance from moving agents
  - moving toward a visible target if a goal is included
- comparison between naive action selection and model-based planning behavior

### Suggested Planning Methods

Keep Phase 2A simple:

- random shooting over short action sequences
- one-step or few-step lookahead
- heuristic score over predicted future observations

Do not start with Monte Carlo Tree Search or complex latent-space planners.

### Completion Criteria

Phase 2A is complete when:

1. A trained world model can be loaded for inference.
2. The planner evaluates multiple candidate actions or short action sequences.
3. Planned behavior measurably improves a chosen objective over a naive baseline in at least one controlled scenario.
4. The visualization can demonstrate planning decisions clearly.

### Fallback Delivery Point

If full planning is too ambitious, the minimum acceptable Phase 2A delivery is:

- one-step model-based action scoring
- a narrow scenario where predicted outcomes are used to avoid a bad action

That still shows the bridge from prediction to decision-making.

## Phase 2B: Small Model-Based RL Extension

Phase 2B may add a small model-based reinforcement learning extension after Phase 2A works.

### Objective

Explore a lightweight RL extension without expanding the project beyond one-month scope.

### Possible Scope

- Dyna-style imagined rollouts for extra training data
- model-assisted policy evaluation
- simple policy learning in the same grid-world using world-model-generated samples

### Constraints

- must reuse Phase 1 and Phase 2A infrastructure
- must remain small enough to explain clearly in a portfolio setting
- must not destabilize the baseline deliverables

### Completion Criteria

Phase 2B is complete if:

1. The RL extension is clearly scoped and isolated.
2. The baseline planner and world model still function unchanged.
3. There is at least one measurable comparison against a simpler non-RL baseline.

### Acceptable Outcome

Phase 2B is optional. If time runs short, the correct choice is to stop after a polished Phase 2A.

## Recommended Weekly Progression

### Week 1

- finalize architecture
- implement environment design
- validate local observation extraction
- establish test scaffolding

### Week 2

- generate rollout datasets
- implement encoder-GRU-decoder baseline
- run first local training smoke tests

### Week 3

- improve training stability
- build polished Pygame visualization
- prepare Colab training workflow

### Week 4

- complete planning baseline
- refine evaluation and demos
- only then consider a small RL extension if time remains

## Explicit Anti-Scope

Do not expand the project with:

- deep multi-agent learning
- transformer-based sequence modeling as the default baseline
- distributed training
- cloud deployment
- large-scale hyperparameter search
- benchmark-chasing complexity

The strongest portfolio result will come from a coherent, finished system.
