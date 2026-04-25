# Development Plan

## 1. Goal and Deadline

**Deadline:** May 4, 2026

The goal is to complete a reproducible course project on **Multi-Agent Reinforcement Learning with Independent PPO and LLM-assisted Guidance on PettingZoo Simple Spread**.

The final project should include:

- A working PettingZoo Simple Spread training pipeline.
- Independent PPO (IPPO) baseline.
- Basic Multi-Agent PPO / MAPPO-style comparison.
- Optional LLM-assisted guidance module.
- Multi-seed experiments.
- Figures, tables, and trajectory visualizations.
- Final report and submission zip.

## 2. Team Role Split

The group has two members. The recommended split is based on parallel work while keeping one person responsible for the core training pipeline.

### Member A: Core RL Pipeline and IPPO

Primary responsibilities:

- Set up repository structure.
- Build and test the PettingZoo Simple Spread wrapper.
- Implement shared utilities for seeding, logging, checkpointing, and metrics.
- Implement IPPO baseline.
- Maintain `src/train.py`, `src/evaluate.py`, and config loading.
- Run smoke tests on MacBook Pro.
- Run or coordinate full training on the RTX 4070 machine.

Main output files:

- `src/envs/simple_spread_wrapper.py`
- `src/algorithms/ippo.py`
- `src/train.py`
- `src/evaluate.py`
- `src/utils.py`
- `configs/ippo.yaml`

### Member B: MAPPO, LLM Guidance, Results, and Report

Primary responsibilities:

- Implement the MAPPO-style centralized critic comparison.
- Implement the optional LLM guidance module.
- Define how LLM output becomes reward shaping, sub-goals, or strategy labels.
- Build plotting and result aggregation scripts.
- Generate learning curves, metric tables, and trajectory visualizations.
- Lead report writing and final submission packaging.

Main output files:

- `src/algorithms/mappo.py`
- `src/llm/guidance.py`
- `src/plot_results.py`
- `configs/mappo.yaml`
- `configs/llm_guidance.yaml`
- `results/`
- `REPORT.pdf`

Both members should review each other's work before final submission.

## 3. Execution Phases

## Phase 1: Repository and Environment Setup

**Target date:** Apr 25

Tasks:

- Create the planned folder structure.
- Create Python virtual environment.
- Install core dependencies.
- Validate PettingZoo Simple Spread can reset, step, and render.
- Decide whether the RTX 4070 machine will use Windows native Python or WSL2 Linux.
- Create `.env.example` for optional LLM API keys.
- Start `README.md` with setup instructions.

Commands to validate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "pettingzoo[mpe]" supersuit "stable-baselines3[extra]" gymnasium numpy torch
```

Expected output:

- Dependencies install successfully.
- A minimal script can create `simple_spread_v3`.
- Team confirms MacBook for development and RTX 4070 machine for full training.

## Phase 2: Simple Spread Wrapper and Smoke Tests

**Target date:** Apr 26

Tasks:

- Implement `src/envs/simple_spread_wrapper.py`.
- Expose observations, actions, rewards, dones, and global state / concatenated observations.
- Add deterministic seed handling.
- Add simple metric calculation:
  - episode return
  - coverage distance
  - collision count or collision rate
- Run random-policy rollouts.
- Save a small sample log to verify logging format.

Expected output:

- Environment wrapper works for at least 10 random episodes.
- Metrics are saved correctly.
- No hard-coded absolute paths.

## Phase 3: IPPO Baseline

**Target date:** Apr 27-28

Tasks:

- Implement `src/algorithms/ippo.py`.
- Implement actor and critic networks.
- Implement rollout buffer and GAE if not using SB3 directly.
- Implement PPO update:
  - clipped policy objective
  - value loss
  - entropy bonus
  - gradient clipping
- Add config file `configs/ippo.yaml`.
- Connect IPPO to `src/train.py`.
- Run short single-seed training on MacBook Pro.

Suggested smoke test:

```bash
python src/train.py --config configs/ippo.yaml --seed 0
```

Expected output:

- Training loop runs end-to-end.
- Logs contain episode return, coverage distance, and collision rate.
- At least one checkpoint can be saved and loaded.
- IPPO curve does not need to be good yet, but it should not crash.

## Phase 4: MAPPO-Style Baseline and LLM Guidance

**Target date:** Apr 29-30

### MAPPO-Style Baseline

Tasks:

- Implement `src/algorithms/mappo.py`.
- Reuse IPPO actor where possible.
- Add centralized critic input based on global state or concatenated local observations.
- Use centralized values for advantage estimation.
- Add config file `configs/mappo.yaml`.
- Run short single-seed training and compare logs with IPPO.

Expected output:

- MAPPO-style training runs end-to-end.
- The method can be evaluated with the same metrics as IPPO.

### LLM Guidance

Tasks:

- Implement `src/llm/guidance.py`.
- Support OpenAI or Anthropic API through `.env`.
- Add a fallback mode when no API key is available.
- Define prompt template for Simple Spread.
- Convert LLM output into one of the following:
  - agent-landmark assignment
  - auxiliary reward shaping
  - curriculum mode
  - strategy label for analysis
- Add config file `configs/llm_guidance.yaml`.

Expected output:

- LLM guidance can be enabled or disabled from config.
- LLM is called only at episode reset or fixed intervals.
- IPPO and MAPPO still run without API keys.

## Phase 5: Multi-Seed Experiments

**Target date:** May 1

Run full experiments on the RTX 4070 machine.

Methods:

- IPPO
- MAPPO-style PPO
- LLM-assisted IPPO

Seeds:

- `0`
- `1`
- `2`

Suggested commands:

```bash
python src/train.py --config configs/ippo.yaml --seed 0
python src/train.py --config configs/ippo.yaml --seed 1
python src/train.py --config configs/ippo.yaml --seed 2

python src/train.py --config configs/mappo.yaml --seed 0
python src/train.py --config configs/mappo.yaml --seed 1
python src/train.py --config configs/mappo.yaml --seed 2

python src/train.py --config configs/llm_guidance.yaml --seed 0
python src/train.py --config configs/llm_guidance.yaml --seed 1
python src/train.py --config configs/llm_guidance.yaml --seed 2
```

Expected output:

- One log file per method and seed.
- Best checkpoint for each method.
- Evaluation summaries saved under `results/`.

If time is limited, prioritize:

1. IPPO with 3 seeds.
2. MAPPO-style PPO with 3 seeds.
3. LLM-assisted IPPO with at least 1-3 seeds depending on API access and runtime.

## Phase 6: Evaluation and Visualization

**Target date:** May 2

Tasks:

- Run `src/evaluate.py` on saved checkpoints.
- Aggregate results across seeds.
- Generate final figures with `src/plot_results.py`.
- Save output under `results/`.

Required figures:

- `results/learning_curves.png`
- `results/collision_rate.png`
- `results/coverage_distance.png`
- `results/comparison_table.png`
- representative trajectory plots under `results/trajectories/`

Expected output:

- Learning curves show mean and standard deviation across seeds.
- Tables compare IPPO, MAPPO-style PPO, and LLM-assisted IPPO.
- Trajectory plots show qualitative cooperation or failure cases.

## Phase 7: Report and Submission Packaging

**Target date:** May 3

Tasks:

- Fill in `PROJECT_REPORT_DRAFT.md` with actual results.
- Export final `REPORT.pdf`.
- Complete `README.md`.
- Finalize `requirements.txt`.
- Check `SUBMISSION_CHECKLIST.md`.
- Remove secrets and unnecessary large files.
- Create final zip package.

Expected final zip structure:

```text
marl-ppo-llm-simple-spread.zip
├── README.md
├── requirements.txt
├── REPORT.pdf
├── SUBMISSION_CHECKLIST.md
├── src/
├── configs/
├── results/
├── logs/
└── checkpoints/
```

## Phase 8: Final Verification

**Target date:** May 4

Tasks:

- Test installation instructions in a clean environment if time allows.
- Run a minimal smoke test from the submitted zip.
- Confirm all report figures are generated from code.
- Confirm all team members are listed in the report.
- Confirm `.env` and API keys are not included.
- Submit before the deadline.

## 4. Daily Schedule

| Date | Main Goal | Owner | Concrete Output |
| --- | --- | --- | --- |
| Apr 25 | Setup and documentation alignment | Both | Folder structure, environment installed, docs aligned |
| Apr 26 | Environment wrapper | Member A | Random rollout and metric logging work |
| Apr 27 | IPPO first version | Member A | IPPO training loop runs |
| Apr 28 | IPPO validation and logging | Member A | Single-seed IPPO curve and checkpoint |
| Apr 29 | MAPPO-style method | Member B | Centralized critic training loop runs |
| Apr 30 | LLM guidance module | Member B | Optional guidance works with fallback |
| May 1 | Full experiments | Both | Multi-seed logs and checkpoints |
| May 2 | Evaluation and plots | Member B | Final result figures and tables |
| May 3 | Report and packaging | Both | Draft final report and zip contents |
| May 4 | Final check and submission | Both | Submitted project package |

## 5. Machine Usage Plan

Use the MacBook Pro for development:

- Write and review code.
- Run short smoke tests.
- Debug LLM API calls.
- Generate report text.

Use the RTX 4070 gaming laptop / desktop for full experiments:

- Run multi-seed training.
- Save checkpoints and logs.
- Run batch evaluation.

SSH recommendation:

- SSH is not required if training is launched directly on the gaming laptop.
- SSH is useful if the team wants to control the gaming laptop from the MacBook.
- If SSH is used, run long jobs in `tmux` so training continues after disconnecting.

## 6. Definition of Done

The project is considered complete when:

- `python src/train.py --config configs/ippo.yaml` works.
- `python src/train.py --config configs/mappo.yaml` works.
- `python src/train.py --config configs/llm_guidance.yaml` works or clearly falls back when no API key exists.
- At least 3 seeds are evaluated for the main methods.
- `results/` contains final curves, tables, and trajectory visualizations.
- `REPORT.pdf` contains method descriptions, experiment settings, results, and discussion.
- `README.md` explains how to reproduce experiments.
- `requirements.txt` is included.
- No API keys or `.env` file are submitted.

## 7. Priority Rules If Time Is Tight

If the schedule becomes too tight, use this priority order:

1. Make IPPO correct and reproducible.
2. Add MAPPO-style comparison.
3. Generate clean figures and write a strong report.
4. Add LLM-assisted guidance as an extension.
5. Add extra videos, notebooks, or ablation studies only if core results are finished.

The grading criteria emphasize technical correctness, experimental rigor, and report quality. A stable IPPO plus MAPPO comparison with clear multi-seed results is more valuable than an overly complex but unstable implementation.
