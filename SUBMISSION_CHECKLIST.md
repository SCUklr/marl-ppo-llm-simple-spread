# IERG 5350 Course Project — Submission Checklist

> **Project Title:** Comparing REINFORCE and PPO on Multi-Agent Cooperative Control (Simple Spread)  
> **Deadline:** May 4, 2026  
> **Format:** `.zip` archive uploaded to course submission portal

---

## 1. Program Codes (Source Code)

All code required to reproduce experiments and regenerate report figures.

| Item | Status | Filename / Path | Notes |
|------|--------|-----------------|-------|
| REINFORCE training script | ☐ | `src/train_reinforce.py` | |
| PPO training script | ☐ | `src/train_ppo.py` | |
| Neural network models | ☐ | `src/models.py` | Actor + Critic architectures |
| Utility functions | ☐ | `src/utils.py` | GAE, replay buffer, logging, etc. |
| Environment wrapper (if any) | ☐ | `src/env_wrapper.py` | MPE Simple Spread wrapper |
| Configuration files | ☐ | `configs/*.yaml` | Hyperparameters for both algorithms |
| **README.md** | ☐ | `README.md` | Installation & running instructions |
| **requirements.txt** | ☐ | `requirements.txt` | Python dependencies with versions |
| Jupyter Notebooks (optional) | ☐ | `notebooks/*.ipynb` | For analysis / visualization |

**Requirements for Code:**
- [ ] Code is well-commented and follows consistent style
- [ ] Running `README.md` instructions reproduces all experiments
- [ ] All random seeds are set for reproducibility
- [ ] No hard-coded absolute paths

---

## 2. Project Report (PDF / Markdown)

Approximately **4–8 pages** (excluding references). Must include:

| Section | Required Content | Page Estimate | Status |
|---------|------------------|---------------|--------|
| **1. Introduction & Problem Setting** | Motivation, why Topic #5, problem description, MDP formulation (State, Action, Reward, Transition) | 0.5–1 pg | ☐ |
| **2. Approach / Methodology** | REINFORCE and PPO descriptions, neural network architecture, why these methods, any custom modifications | 1–2 pg | ☐ |
| **3. Experiment Settings** | Hyperparameters, training episodes, evaluation protocol, metrics, hardware, random seeds | 0.5–1 pg | ☐ |
| **4. Results** | Learning curves (mean ± std over seeds), performance tables, trajectory visualizations | 1–2 pg | ☐ |
| **5. Findings & Discussion** | Analysis of results, failure modes, algorithm comparison, limitations | 1–2 pg | ☐ |
| **References** | All cited papers and resources | — | ☐ |

**Report Quality Checklist:**
- [ ] Figures are high-resolution, properly labeled (axes, legend, caption)
- [ ] Tables include units and clear column headers
- [ ] Results averaged over **multiple random seeds** (≥3)
- [ ] Clear comparison between REINFORCE and PPO
- [ ] Proper academic writing style

---

## 3. Trained Model Checkpoints (Optional but Recommended)

| Item | Status | Filename | Notes |
|------|--------|----------|-------|
| Best REINFORCE checkpoint | ☐ | `checkpoints/reinforce_best.pt` | |
| Best PPO checkpoint | ☐ | `checkpoints/ppo_best.pt` | |
| Training logs | ☐ | `logs/*.csv` or TensorBoard event files | For reproducibility |

---

## 4. Final Deliverable Structure

The submitted `.zip` file should have the following structure:

```
marl-ppo-llm-simple-spread.zip
├── README.md
├── requirements.txt
├── REPORT.pdf                    # Final project report
├── SUBMISSION_CHECKLIST.md       # This file
├── src/
│   ├── train_reinforce.py
│   ├── train_ppo.py
│   ├── models.py
│   ├── utils.py
│   └── env_wrapper.py
├── configs/
│   ├── reinforce_config.yaml
│   └── ppo_config.yaml
├── checkpoints/                  # Optional
│   ├── reinforce_best.pt
│   └── ppo_best.pt
├── logs/                         # Optional
│   ├── reinforce_logs.csv
│   └── ppo_logs.csv
├── results/                      # Figures used in report
│   ├── learning_curves.png
│   ├── comparison_table.png
│   └── trajectories/
└── notebooks/                    # Optional
    └── analysis.ipynb
```

---

## 5. Evaluation Criteria Self-Check

Before submitting, verify your project meets the grading rubric:

| Criterion | Weight | Self-Assessment | Notes |
|-----------|--------|-----------------|-------|
| **Technical Correctness** | 30% | ☐ | Are REINFORCE and PPO implemented correctly? Is the MDP formulation sound? |
| **Experimental Rigor** | 30% | ☐ | Multiple seeds? Proper baselines? Statistical significance? |
| **Quality of Report** | 30% | ☐ | Well-structured? Clear figures? Proper analysis? |
| **Originality & Effort** | 10% | ☐ | Beyond bare minimum? Thorough comparison or unique extension? |

---

## 6. Action Items Before Submission

- [ ] Run full training pipeline end-to-end on a clean environment
- [ ] Verify `README.md` instructions work on a fresh clone
- [ ] Generate all report figures from code (not hand-drawn)
- [ ] Proofread report for clarity and grammar
- [ ] Check that all team members are listed on the report
- [ ] Ensure `.zip` file size is reasonable (< 100 MB preferably)
- [ ] Submit before **May 4, 2026** deadline
