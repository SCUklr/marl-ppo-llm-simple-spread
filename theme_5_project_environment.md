# Theme 5 Project Extraction and Environment Setup

## 1. PDF Extracted Topic

Source: `rl_project.pdf`, Theme 5.

### Title

Multi-Agent Reinforcement Learning with Independent PPO and LLM-assisted Guidance on PettingZoo Simple Spread (Multi-Agent + LLM)

### Project Description

This project sets up PettingZoo Simple Spread for multi-agent collaboration, trains Independent PPO agents, compares with basic Multi-Agent PPO, and integrates a lightweight LLM for high-level natural language guidance. It analyzes credit assignment, collaboration performance, and visualizes emergent cooperation patterns through multi-seed experiments.

### Keywords

Multi-Agent RL, PettingZoo, Simple Spread, Independent PPO, MAPPO, LLM-assisted RL, Hierarchical RL, Credit Assignment, Collaboration, Stable-Baselines3

## 2. Project Understanding

这个主题的核心是做一个多智能体强化学习实验：在 PettingZoo 的 Simple Spread 环境中，让多个 agent 学会协作覆盖 landmark。基础方案是 Independent PPO，也就是每个智能体各自用 PPO 学习；进阶对比是 MAPPO 或简化版 centralized critic PPO；LLM 部分不需要训练大模型，而是作为高层指导模块，用自然语言生成子目标、策略提示或奖励 shaping 规则。

建议最终实验至少覆盖以下内容：

- Independent PPO baseline：每个 agent 独立学习。
- MAPPO / basic Multi-Agent PPO：作为多智能体 PPO 对比方法。
- LLM-assisted guidance：在 episode 开始或关键决策点调用 LLM，避免每一步都调用导致训练过慢。
- Multi-seed evaluation：至少 3 个随机种子，报告均值和方差。
- Visualization：学习曲线、agent 轨迹、协作覆盖效果、collision rate。

## 3. Recommended Software Environment

### 3.1 Operating System

推荐两套可用环境：

| Machine | Recommended Use | Notes |
| --- | --- | --- |
| macOS | 代码开发、单 seed 快速测试、LLM API 调试 | Simple Spread 不依赖 MuJoCo，macOS 可直接运行 |
| Windows / Linux + NVIDIA GPU | 完整训练、多 seed 实验、批量跑结果 | RTX 4070 足够运行该项目 |

如果只在 Mac 上完成项目也可以，但完整多 seed 训练会更慢。Simple Spread 的模型规模不大，CPU 也能跑，只是训练时间更长。

### 3.2 Python Version

推荐使用：

- Python 3.10 或 Python 3.11

不建议使用过新的 Python 版本作为主环境，因为 PettingZoo、SuperSuit、Stable-Baselines3、PyTorch 之间的版本兼容性更容易出问题。

### 3.3 Core RL Dependencies

| Package | Recommended Version | Purpose |
| --- | --- | --- |
| `pettingzoo[mpe]` | >= 1.24 | Simple Spread 多智能体环境 |
| `supersuit` | >= 3.9 | PettingZoo 环境包装，方便接入 Gymnasium / SB3 |
| `stable-baselines3[extra]` | >= 2.3 | PPO baseline 和训练工具 |
| `torch` | >= 2.1 | 神经网络后端 |
| `gymnasium` | >= 0.29 | RL 环境接口 |
| `numpy` | >= 1.24 | 数值计算 |

### 3.4 LLM Integration Dependencies

如果使用 OpenAI API：

| Package | Purpose |
| --- | --- |
| `openai` | 调用 GPT 系列模型作为 high-level guidance |
| `python-dotenv` | 从 `.env` 安全读取 API key |

如果使用 Anthropic Claude API：

| Package | Purpose |
| --- | --- |
| `anthropic` | 调用 Claude 模型作为 high-level guidance |
| `python-dotenv` | 从 `.env` 安全读取 API key |

LLM 部分推荐做成可选模块：没有 API key 时仍然可以跑 Independent PPO 和 MAPPO；有 API key 时再开启 LLM-assisted guidance。

### 3.5 Experiment Tracking and Visualization

| Package / Tool | Purpose |
| --- | --- |
| `tensorboard` | 查看训练曲线 |
| `wandb` | 可选，用于实验追踪和多 seed 对比 |
| `matplotlib` | 绘制学习曲线和结果图 |
| `seaborn` | 更美观的统计图 |
| `pandas` | 聚合不同 seed 的结果 |
| `imageio` | 保存 episode 视频或 GIF |
| `opencv-python` | 可选，处理视频帧 |
| `jupyter` | 可选，用 notebook 做结果分析 |

### 3.6 Development Tools

| Tool | Purpose |
| --- | --- |
| Git | 版本管理 |
| VS Code / Cursor | 开发环境 |
| `requirements.txt` | 固定依赖版本，便于提交和复现 |
| `.env` | 保存 LLM API key，不要提交到 Git |

## 4. Installation Commands

### 4.1 Create Virtual Environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4.2 Install Core Dependencies

```bash
pip install "pettingzoo[mpe]" supersuit "stable-baselines3[extra]" gymnasium numpy
```

### 4.3 Install PyTorch

For macOS or CPU-only:

```bash
pip install torch
```

For NVIDIA GPU, install the CUDA build according to the official PyTorch selector. For an RTX 4070 machine, CUDA 12.1 wheels are usually suitable:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4.4 Install LLM and Visualization Dependencies

```bash
pip install openai anthropic python-dotenv
pip install tensorboard wandb matplotlib seaborn pandas imageio opencv-python jupyter
```

### 4.5 Generate `requirements.txt`

After the environment works:

```bash
pip freeze > requirements.txt
```

## 5. Environment Variables

Create a `.env` file in the project root if using an LLM API:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

只需要配置实际使用的 provider。`.env` 不应该提交到 Git。

## 6. Recommended Project Structure

```text
marl-ppo-llm-simple-spread/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── ippo.yaml
│   ├── mappo.yaml
│   └── llm_guidance.yaml
├── src/
│   ├── envs/
│   │   └── simple_spread_wrapper.py
│   ├── algorithms/
│   │   ├── ippo.py
│   │   └── mappo.py
│   ├── llm/
│   │   └── guidance.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_results.py
├── logs/
├── checkpoints/
├── results/
└── notebooks/
```

## 7. Notes and Risks

- PettingZoo 的 MPE 环境需要安装 `pettingzoo[mpe]`，只装 `pettingzoo` 可能缺少 Simple Spread 相关依赖。
- Stable-Baselines3 原生面向单智能体 Gymnasium 环境；Independent PPO 需要把每个 agent 的 observation/action 拆开处理，或通过 wrapper 转换。
- MAPPO 可能需要自己实现 centralized critic，SB3 不能直接完整支持标准 MAPPO。
- LLM 调用不要放在每个 timestep，否则 API 延迟和费用会明显影响训练。建议只在 episode reset、固定 interval 或评估阶段调用。
- 最终报告需要说明 LLM guidance 如何被转成 RL 可用信号，例如 reward shaping、sub-goal selection、prompt-generated strategy label 或 curriculum schedule。

