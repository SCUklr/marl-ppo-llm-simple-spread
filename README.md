# MARL PPO LLM Simple Spread

This project studies multi-agent reinforcement learning on PettingZoo Simple Spread. The planned methods are Independent PPO (IPPO), a MAPPO-style centralized critic comparison, and optional LLM-assisted high-level guidance.

The current implementation provides the project scaffold, a random-policy smoke test, IPPO training, a MAPPO-style centralized critic baseline, optional heuristic LLM guidance, evaluation, and plotting.

## Setup

Use Python 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the local Python installation cannot bootstrap `pip`, use `uv` instead:

```bash
uv venv --python 3.11 --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

For an RTX 4070 Windows / WSL2 machine, install the CUDA-enabled PyTorch wheel according to the official PyTorch selector if the default `torch` wheel does not detect CUDA.

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

## Evaluation and Plotting

Evaluate a saved checkpoint:

```bash
.venv/bin/python src/evaluate.py \
  --config configs/ippo.yaml \
  --checkpoint checkpoints/ippo_seed_0_best.pt \
  --episodes 10 \
  --output results/ippo_eval.csv
```

Generate plots from training logs:

```bash
.venv/bin/python src/plot_results.py --output_dir results
```

## Project Structure

```text
marl-ppo-llm-simple-spread/
├── configs/
│   ├── random_rollout.yaml
│   ├── ippo.yaml
│   ├── mappo.yaml
│   └── llm_guidance.yaml
├── scripts/
│   └── run_lightweight_experiments.py
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
├── results/
└── notebooks/
```

## Machine Usage

Develop and run short tests on the MacBook Pro. Run full multi-seed experiments on the RTX 4070 laptop if the final training time becomes too long. The project can still be completed locally on the MacBook because Simple Spread uses lightweight MLP policies and does not require MuJoCo.

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

If the key works, the output should show `source=qwen`. If no key is found or the provider fails, it will show `source=heuristic` or `source=heuristic_fallback`.

The lightweight config refreshes guidance every 100 episodes. With 3 seeds x 500 episodes, this means about 15 guidance refreshes for the LLM-guided method. If replaced with a real API provider, expect about 15 API calls for the default lightweight run, or 9 API calls for 3 seeds x 300 episodes.
