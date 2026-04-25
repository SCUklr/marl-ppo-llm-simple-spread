# MARL PPO LLM Simple Spread

This project studies multi-agent reinforcement learning on PettingZoo Simple Spread. The planned methods are Independent PPO (IPPO), a MAPPO-style centralized critic comparison, and optional LLM-assisted high-level guidance.

The current first iteration provides the project scaffold and a random-policy smoke test for validating the environment, metrics, config loading, and logging.

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

## Planned Training Commands

These commands will be enabled in later iterations:

```bash
python src/train.py --config configs/ippo.yaml
python src/train.py --config configs/mappo.yaml
python src/train.py --config configs/llm_guidance.yaml
```

## Project Structure

```text
marl-ppo-llm-simple-spread/
├── configs/
│   └── random_rollout.yaml
├── src/
│   ├── envs/
│   │   └── simple_spread_wrapper.py
│   ├── algorithms/
│   ├── llm/
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

LLM guidance is optional. If used, copy `.env.example` to `.env` and fill in the provider key. Do not commit `.env`.
