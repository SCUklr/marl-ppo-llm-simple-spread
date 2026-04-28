# Software & Hardware Resource Requirements

**Project**: Multi-Agent Reinforcement Learning with Independent PPO and LLM-assisted Guidance on PettingZoo Simple Spread  
**Course**: IERG5350 Reinforcement Learning in Practice  
**Deadline**: May 4, 2026

---

## Hardware Resources

### Primary Training Machine — Desktop (RTX 4070)
| Component | Spec | Role |
|-----------|------|------|
| GPU | NVIDIA RTX 4070 (12 GB VRAM) | PPO policy network training, multi-seed parallel experiments |
| OS | Windows / Linux recommended | PyTorch CUDA support |
| RAM | ≥ 16 GB recommended | Environment rollout buffers for multiple agents |
| Storage | ≥ 20 GB free | Model checkpoints, logs, video recordings |

> The RTX 4070 is sufficient for this project. The PPO networks for Simple Spread are small MLPs; GPU provides a ~3–5x speedup over CPU for multi-seed parallel runs.

### Secondary Development Machine — MacBook Pro
| Component | Role |
|-----------|------|
| CPU (Apple Silicon or Intel) | Code development, API testing, lightweight single-seed runs |
| macOS | Compatible with PettingZoo, Gymnasium, PyTorch (MPS backend available for Apple Silicon) |
| Internet | Required for LLM API calls |

> MuJoCo-free environments like Simple Spread run on macOS without extra setup. Use the MacBook for development and the desktop for full multi-seed training runs.

### Network
- Stable internet connection required on whichever machine runs the LLM integration, as all LLM calls go through external API (no local deployment).
- Estimated API latency: ~0.5–2s per call; design the guidance integration to call the LLM only at episode boundaries or key decision points to avoid bottlenecks.

---

## Software Resources

### Core Environment & RL Stack
| Package | Version (recommended) | Purpose |
|---------|-----------------------|---------|
| Python | 3.10 or 3.11 | Base language |
| PettingZoo | ≥ 1.24 | Multi-agent environment (Simple Spread) |
| SuperSuit | ≥ 3.9 | PettingZoo → Gymnasium wrapper for SB3 compatibility |
| Stable-Baselines3 | ≥ 2.3 | PPO implementation (Independent PPO baseline) |
| PyTorch | ≥ 2.1 (CUDA 12.x build on desktop) | Neural network backend |
| Gymnasium | ≥ 0.29 | Standard RL environment interface |
| NumPy | ≥ 1.24 | Array operations |

### LLM Integration (API-based)
| Package | Purpose |
|---------|---------|
| `openai` Python SDK | If using GPT-4o / GPT-3.5 as the LLM guide |
| `anthropic` Python SDK | If using Claude as the LLM guide |
| `python-dotenv` | Securely load API keys from `.env` file |

> **No local model deployment needed.** All LLM inference goes through the provider API. The LLM acts as a high-level guidance module (e.g., generating natural language hints that are encoded as reward shaping signals or sub-goal instructions at episode resets).

### Experiment Tracking & Visualization
| Package | Purpose |
|---------|---------|
| TensorBoard or Weights & Biases (`wandb`) | Training curve logging, multi-seed aggregation |
| Matplotlib / Seaborn | Learning curve plots for the report |
| Pandas | Results aggregation across seeds |
| OpenCV (`opencv-python`) or `imageio` | Video recording of agent behavior |

### Development Tools
| Tool | Purpose |
|------|---------|
| Jupyter Notebook | Exploratory analysis, result visualization |
| `requirements.txt` | Dependency pinning for reproducibility (required by project spec) |
| Git + GitHub | Version control and submission |

---

## Installation Notes

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Core dependencies
pip install pettingzoo[mpe] supersuit stable-baselines3[extra] torch

# LLM API (choose one or both)
pip install openai anthropic python-dotenv

# Visualization & tracking
pip install wandb matplotlib seaborn pandas opencv-python jupyter

# Pin versions for reproducibility
pip freeze > requirements.txt
```

> On the RTX 4070 desktop, install the CUDA build of PyTorch:  
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## Resource Summary

| Dimension | What We Have | Sufficiency |
|-----------|-------------|-------------|
| GPU training | RTX 4070 (12 GB) | Sufficient — Simple Spread networks are small |
| CPU development | MacBook Pro | Sufficient for dev and single-seed tests |
| LLM inference | API (OpenAI / Anthropic) | Sufficient — no local GPU needed for LLM |
| API cost estimate | ~$1–5 total for multi-seed experiments (if calls limited to episode boundaries) | Low cost |
| Storage | 20 GB recommended | Covers logs, checkpoints, videos |
