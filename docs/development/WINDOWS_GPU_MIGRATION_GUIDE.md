# Windows GPU Migration Guide

## 1. What This Project Is Doing

This project studies multi-agent reinforcement learning on PettingZoo `simple_spread_v3`.

The current experimental design has three main methods:

- IPPO as the main baseline.
- A MAPPO-style variant with a centralized critic as the comparison method.
- An optional LLM-guided IPPO variant for lightweight strategic guidance or reward shaping.

The core workflow is:

1. Train with `src/train.py` using YAML configs.
2. Save logs to `logs/` and checkpoints to `checkpoints/`.
3. Evaluate with `src/evaluate.py`.
4. Generate figures with `src/plot_results.py`.

## 2. Current Progress

At the current stage, the project is no longer just an idea or empty scaffold. The following pieces already exist in the repository:

- Environment wrapper for Simple Spread.
- Shared training utilities for seeding, YAML loading, and CSV logging.
- IPPO training pipeline.
- MAPPO-style training pipeline.
- Optional LLM guidance module and config.
- Evaluation script.
- Plotting script.
- Lightweight multi-seed experiment runner.
- Project documentation under `docs/`.

In practical terms, the project has reached the "can run experiments" stage, but it is still in the experimentation and final-results stage rather than the fully packaged final-submission stage.

That means:

- The code structure is already in place.
- Training configs already exist.
- The main work left is running stable experiments, collecting results, making figures, and polishing the final report and submission package.

## 3. Recommended Migration Strategy

The recommended approach is:

1. Keep GitHub as the source of truth for code.
2. Use the Windows gaming laptop as the main training machine.
3. Run the project inside WSL2 Ubuntu on that laptop instead of native Windows Python.

This is the best fit for the current repository because the project already uses Linux-style commands and paths in the documentation and scripts, for example `.venv/bin/python`.

Using WSL2 means:

- Fewer code changes.
- Better compatibility with the current README commands.
- Easier use of PyTorch, PettingZoo, and Python tooling.
- Easier future remote development if you later use SSH, Tailscale, or Cursor remote workflows.

## 4. What Needs To Change For the Windows Laptop

### 4.1 Recommended choice: use WSL2

If you use WSL2 Ubuntu, almost no project code needs to change.

The current code already supports GPU selection automatically. The training configs use:

```yaml
training:
  device: auto
```

The device selection logic already prefers CUDA when available, then MPS, then CPU. So on the gaming laptop, if PyTorch with CUDA is installed correctly, training should automatically use the NVIDIA GPU.

So for WSL2, the main "changes" are environment-level rather than code-level:

- Install WSL2 and Ubuntu.
- Install NVIDIA drivers on Windows.
- Make sure CUDA-enabled PyTorch is installed inside WSL2.
- Clone the repository from GitHub.
- Create a new Python environment on the laptop.
- Reinstall dependencies there.
- Run training from WSL2.

### 4.2 If you insist on native Windows Python

Native Windows is possible, but it is not the smoothest option for this repository.

The main difference is that some commands assume Linux paths such as:

```bash
.venv/bin/python
```

On native Windows, the equivalent is usually:

```powershell
.venv\Scripts\python.exe
```

In particular, `scripts/run_lightweight_experiments.py` defaults to `.venv/bin/python`, so under native Windows you would need to run it with an explicit Python path override, for example:

```powershell
python scripts/run_lightweight_experiments.py --python .venv\Scripts\python.exe --episodes 500 --seeds 0 1 2
```

Because of this, WSL2 is still the recommended setup.

## 5. GitHub-Based Migration Workflow

The intended workflow is:

1. Commit and push the latest code from your current machine to GitHub.
2. On the gaming laptop, clone the repository.
3. Check out the same branch.
4. Build a fresh environment on the laptop.
5. Run experiments on the laptop.
6. Commit and push any code or config updates back to GitHub.
7. Pull the latest branch on your Mac when needed.

Typical workflow:

```bash
git clone <repo-url>
cd marl-ppo-llm-simple-spread
git checkout <branch-name>
```

After that:

```bash
git pull
# run experiments or edit configs
git add .
git commit -m "update experiment settings"
git push
```

This makes the gaming laptop the execution machine while GitHub remains the synchronization layer between devices.

## 6. Laptop Setup Checklist

### 6.1 System setup

Recommended:

- Windows 11
- WSL2
- Ubuntu
- Latest NVIDIA driver

### 6.2 Python environment setup inside WSL2

Inside the cloned project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your WSL image is missing `pip` / `venv`, install `uv` in user space and let it manage Python and the virtual environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install 3.11
uv venv --python 3.11 --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

You can also use the repo helper after `uv` is available:

```bash
python3 scripts/setup_env.py --python 3.11 --backend uv
```

If the default PyTorch wheel does not detect CUDA, install the CUDA-enabled wheel using the official PyTorch install selector, then verify:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

If this prints `True` and shows your NVIDIA GPU, the project should be able to use CUDA automatically.

You can also write a runtime report for the project configs:

```bash
.venv/bin/python scripts/verify_runtime.py
```

Important compatibility note: keep the pinned `pettingzoo[mpe]==1.24.3` dependency from `requirements.txt`. The current code imports `simple_spread_v3` from `pettingzoo.mpe`, and newer `pettingzoo` releases remove that path.

## 7. How To Run the Experiments on the Gaming Laptop

### 7.1 First run a smoke test

```bash
.venv/bin/python src/train.py --config configs/ippo.yaml --episodes 4 --seed 0
.venv/bin/python src/train.py --config configs/mappo.yaml --episodes 4 --seed 0
.venv/bin/python src/train.py --config configs/llm_guidance.yaml --episodes 4 --seed 0
```

Or run the smoke-test helper:

```bash
.venv/bin/python scripts/run_smoke_tests.py
```

Purpose:

- Confirm dependencies are installed correctly.
- Confirm the environment runs.
- Confirm logging and checkpoint writing work.
- Confirm CUDA is being used if available.

### 7.2 Then run the lightweight official experiment suite

```bash
.venv/bin/python scripts/run_lightweight_experiments.py --episodes 500 --seeds 0 1 2
```

This runs:

- IPPO for seeds `0, 1, 2`
- MAPPO for seeds `0, 1, 2`
- LLM-guided IPPO for seeds `0, 1, 2`

### 7.3 If you want a faster rehearsal first

```bash
.venv/bin/python scripts/run_lightweight_experiments.py --episodes 300 --seeds 0 1 2
```

This is a good first full test on the gaming laptop before committing to longer runs.

## 8. What Should Be Adjusted for the Laptop Experiments

You do not need to redesign the project for the gaming laptop. The main adaptation is operational.

Recommended adjustments:

### 8.1 Keep `training.device: auto`

This is already correct and should automatically use CUDA on the gaming laptop.

### 8.2 Use WSL2 paths and commands

Do not try to mix Windows terminal commands with Linux virtual environment paths in the same workflow. Pick one environment and stay consistent. WSL2 is the preferred environment.

### 8.3 Keep logs, checkpoints, and results local to the laptop during training

The current project already writes outputs to:

- `logs/`
- `checkpoints/`
- `results/`

That is fine. These files can be generated on the gaming laptop and selectively committed only when useful.

### 8.4 Do not commit unnecessary large outputs

Avoid pushing all generated artifacts to GitHub, especially if they become large. In general:

- Code, configs, and important small result summaries can be committed.
- Large checkpoints, repeated logs, temporary caches, and bulky figures should be committed only if really needed.

## 9. Suggested Execution Plan on the Gaming Laptop

Recommended order:

1. Clone the GitHub branch to WSL2.
2. Build the environment.
3. Verify CUDA.
4. Run short smoke tests for IPPO, MAPPO, and LLM-guided IPPO.
5. Run the 300-episode suite as a rehearsal.
6. If everything looks correct, run the 500-episode suite.
7. Generate plots.
8. Copy the final small results or commit the useful outputs back to GitHub.

To finish the run package, create a compact artifact summary:

```bash
.venv/bin/python scripts/summarize_results.py
```

## 10. Bottom-Line Recommendation

For this repository, the cleanest plan is:

- Use GitHub to sync code between machines.
- Use the Windows gaming laptop as the main experiment machine.
- Run the project in WSL2 Ubuntu.
- Keep the code mostly unchanged.
- Focus your effort on environment setup and experiment execution rather than rewriting the project for Windows.

In short: this is mainly a deployment and workflow migration, not an algorithm rewrite.
