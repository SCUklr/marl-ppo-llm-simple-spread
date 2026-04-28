# MARL PPO LLM Simple Spread

This project studies cooperative multi-agent reinforcement learning on PettingZoo `simple_spread_v3`. The repository now includes a working experiment pipeline for three methods:

- `IPPO`: decentralized PPO baseline
- `MAPPO-style PPO`: decentralized actors with a centralized critic
- `LLM-guided IPPO`: IPPO with optional Qwen-based high-level guidance

The current repository state is no longer a scaffold. It already contains:

- a working Simple Spread environment wrapper
- config-driven training, evaluation, and plotting
- logging and checkpoint saving
- multi-seed lightweight experiments (`3 methods x 3 seeds x 500 episodes`)
- independent evaluation CSVs for saved checkpoints
- summary figures and artifact reports under `results/`

## Setup

Use Python 3.10 or 3.11.

Recommended environment:

- `WSL2 Ubuntu` on Windows for the smoothest reproduction
- native Linux also works
- native Windows is supported for TA checking, but replace `.venv/bin/python` with `.venv\Scripts\python.exe`

If you are checking this repository on Windows PowerShell, the command mapping is:

```text
WSL / Linux:    .venv/bin/python
Windows:        .venv\Scripts\python.exe
```

## Quick Reproduction

For a fast end-to-end check, follow this order:

1. Create the environment and install dependencies.
2. Run `scripts/verify_runtime.py`.
3. Run `scripts/run_smoke_tests.py`.
4. Run the lightweight experiment suite.
5. Regenerate plots and the markdown summary.

Minimal reproduction commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/python scripts/verify_runtime.py
.venv/bin/python scripts/run_smoke_tests.py
.venv/bin/python scripts/run_lightweight_experiments.py --episodes 500 --seeds 0 1 2
.venv/bin/python src/plot_results.py --latest_run_only --output_dir results
.venv/bin/python scripts/summarize_results.py --latest_run_only
```

Equivalent PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_runtime.py
.\.venv\Scripts\python.exe scripts\run_smoke_tests.py
.\.venv\Scripts\python.exe scripts\run_lightweight_experiments.py --episodes 500 --seeds 0 1 2
.\.venv\Scripts\python.exe src\plot_results.py --latest_run_only --output_dir results
.\.venv\Scripts\python.exe scripts\summarize_results.py --latest_run_only
```

You can also use the bootstrap helper to create the environment and run a runtime check:

```bash
python3 scripts/setup_env.py --python python3
```

If the local Python installation cannot bootstrap `pip`, use `uv` instead:

```bash
uv venv --python 3.11 --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

For an RTX 4070 Windows / WSL2 machine, install the CUDA-enabled PyTorch wheel according to the official PyTorch selector if the default `torch` wheel does not detect CUDA.

The project imports `simple_spread_v3` from `pettingzoo.mpe`, so keep the pinned `pettingzoo[mpe]==1.24.3` dependency from `requirements.txt`. Newer `pettingzoo` releases remove that import path and will break smoke tests.

Verify the runtime device selection before long runs:

```bash
.venv/bin/python scripts/verify_runtime.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\verify_runtime.py
```

## Smoke Test

Run a short random rollout:

```bash
.venv/bin/python src/train.py --config configs/random_rollout.yaml
```

Override the seed or episode count:

```bash
.venv/bin/python src/train.py --config configs/random_rollout.yaml --seed 0 --episodes 5
```

Expected output:

- Summary metrics printed in the terminal.
- A CSV log saved to `logs/random_rollout.csv`.

You can also run the combined smoke-test helper:

```bash
.venv/bin/python scripts/run_smoke_tests.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_smoke_tests.py
```

This helper runs:

- runtime / CUDA verification
- LLM provider smoke test
- random rollout
- short IPPO / MAPPO / LLM-guided training runs

## Training Commands

Run short smoke tests first:

```bash
.venv/bin/python src/train.py --config configs/ippo.yaml --episodes 4 --seed 0
.venv/bin/python src/train.py --config configs/mappo.yaml --episodes 4 --seed 0
.venv/bin/python src/train.py --config configs/llm_guidance.yaml --episodes 4 --seed 0
```

Run the lightweight official experiment suite:

```bash
.venv/bin/python scripts/run_lightweight_experiments.py --episodes 500 --seeds 0 1 2
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_lightweight_experiments.py --episodes 500 --seeds 0 1 2
```

This runs 3 methods x 3 seeds:

- IPPO: seeds 0, 1, 2
- MAPPO-style PPO: seeds 0, 1, 2
- LLM-guided IPPO: seeds 0, 1, 2

For a faster rehearsal, use fewer episodes:

```bash
.venv/bin/python scripts/run_lightweight_experiments.py --episodes 300 --seeds 0 1 2
```

Single-seed runs are also supported:

```bash
.venv/bin/python src/train.py --config configs/ippo.yaml --episodes 500 --seed 0
.venv/bin/python src/train.py --config configs/mappo.yaml --episodes 500 --seed 0
.venv/bin/python src/train.py --config configs/llm_guidance.yaml --episodes 500 --seed 0
```

## Current Result Snapshot

The latest lightweight comparison table is stored at `results/comparison_table.csv`:

```text
method,final_return,coverage_distance,collision_rate
ippo,-78.25060071097664,0.746018723398447,0.035386666666666663
llm_guided_ippo,-78.22829274665841,0.7454045019745826,0.03562666666666667
mappo,-78.33457561161909,0.7485137739082177,0.03442666666666667
```

Interpretation:

- all three methods run successfully under the same lightweight setting
- `llm_guided_ippo` is slightly better on return / coverage
- `mappo` is slightly better on collision rate
- the differences are small, so the current results support a "methods are comparable" conclusion rather than a strong winner claim

## Evaluation and Plotting

Evaluate a saved checkpoint:

```bash
.venv/bin/python src/evaluate.py \
  --config configs/ippo.yaml \
  --checkpoint checkpoints/ippo_seed_0_best.pt \
  --episodes 10 \
  --seed 10000 \
  --output results/ippo_seed_0_eval.csv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --config configs\ippo.yaml --checkpoint checkpoints\ippo_seed_0_best.pt --episodes 10 --seed 10000 --output results\ippo_seed_0_eval.csv
```

Generate plots from training logs:

```bash
.venv/bin/python src/plot_results.py --latest_run_only --output_dir results
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe src\plot_results.py --latest_run_only --output_dir results
```

Generate a compact markdown summary of the available artifacts:

```bash
.venv/bin/python scripts/summarize_results.py --latest_run_only
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_results.py --latest_run_only
```

Files that should appear after reproduction:

- training logs in `logs/`
- best checkpoints in `checkpoints/`
- figures in `results/`
- summary table in `results/comparison_table.csv`
- artifact summary in `results/experiment_summary.md`

The repository already contains:

- 9 training logs under `logs/`
- 9 best checkpoints under `checkpoints/`
- 9 independent evaluation CSVs under `results/`
- summary figures and `results/experiment_summary.md`

## Documentation

Supporting documents are grouped under `docs/`:

- `docs/development/TECHNICAL_DOCUMENTATION.md`
- `docs/development/WINDOWS_GPU_MIGRATION_GUIDE.md`
- `docs/submission/SUBMISSION_CHECKLIST.md`

## Project Structure

```text
marl-ppo-llm-simple-spread/
├── configs/
│   ├── random_rollout.yaml
│   ├── ippo.yaml
│   ├── mappo.yaml
│   └── llm_guidance.yaml
├── docs/
│   ├── development/
│   └── submission/
├── scripts/
│   ├── run_lightweight_experiments.py
│   ├── run_smoke_tests.py
│   ├── setup_env.py
│   ├── summarize_results.py
│   ├── test_llm_provider.py
│   └── verify_runtime.py
├── src/
│   ├── envs/
│   │   └── simple_spread_wrapper.py
│   ├── algorithms/
│   │   ├── common.py
│   │   ├── ippo.py
│   │   └── mappo.py
│   ├── llm/
│   │   └── guidance.py
│   ├── evaluate.py
│   ├── plot_results.py
│   ├── train.py
│   └── utils.py
├── logs/
├── checkpoints/
└── results/
```

## Machine Usage

The current validated lightweight runs were executed on `WSL2 Ubuntu` with an `RTX 4070 Laptop GPU`. Development and smoke tests can also be done on CPU-only environments, but the checked-in runtime summary reflects the WSL2 + CUDA setup in `results/runtime_check.json`.

## LLM API Keys

LLM guidance is optional. The current `configs/llm_guidance.yaml` is configured for Qwen / DashScope OpenAI-compatible mode, but it safely falls back to a local heuristic if `LLM_API_KEY` is not set.

To use Qwen, copy `.env.example` to `.env` and fill in:

```bash
LLM_API_KEY=your_qwen_key_here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

Do not commit `.env`.

Test the provider before training:

```bash
.venv/bin/python scripts/test_llm_provider.py --config configs/llm_guidance.yaml
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\test_llm_provider.py --config configs\llm_guidance.yaml
```

If the key works, the output should show `source=qwen`. If no key is found or the provider fails, it will show `source=heuristic` or `source=heuristic_fallback`.

The lightweight config refreshes guidance every 100 episodes. With `3 seeds x 500 episodes`, this means about `15` guidance refreshes for the LLM-guided method. The implementation therefore remains an RL training pipeline with sparse high-level guidance rather than an LLM-at-every-step controller.
