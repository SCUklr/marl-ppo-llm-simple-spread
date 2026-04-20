# IERG 5350 Course Project Technical Report — Draft

> **Selected Topic:** #5 — Implementing and Comparing REINFORCE and Proximal Policy Optimization (PPO) on a Control Task  
> **Environment:** Multi-Agent Particle Environment (MPE) – Simple Spread  
> **Team:** [Your Names]  
> **Date:** [Date]

---

## 1. Introduction & Problem Setting

### 1.1 Motivation
Multi-agent coordination is a fundamental challenge in reinforcement learning. In this project, we select **Topic #5** from the course project list: *"Implementing and comparing REINFORCE and Proximal Policy Optimization (PPO) on a control task."* Instead of a single-agent classic control task, we extend the scope to a **multi-agent cooperative control task** — the **Simple Spread** scenario from the Multi-Agent Particle Environment (MPE) — to investigate how policy-gradient methods scale to decentralized decision-making.

### 1.2 MDP Formulation
We formulate the multi-agent task as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**:

| Component | Description |
|-----------|-------------|
| **State space** $\mathcal{S}$ | Global state: positions and velocities of all agents and landmarks |
| **Observation space** $\mathcal{O}_i$ | Each agent observes its own velocity + relative positions of other agents and landmarks (partial observability) |
| **Action space** $\mathcal{A}_i$ | Continuous 2D movement vector per agent |
| **Reward function** $R$ | Negative sum of minimum distances from each landmark to its nearest agent + collision penalty between agents |
| **Transition dynamics** $\mathcal{P}$ | Deterministic physics-driven updates in the MPE simulator |

### 1.3 Research Questions
- Can REINFORCE and PPO learn effective cooperative policies in the Simple Spread environment?
- How do sample efficiency and final performance compare between the two algorithms?
- What are the failure modes (e.g., agent collision, landmark coverage imbalance)?

---

## 2. Approach / Methodology

### 2.1 Algorithms

#### 2.1.1 REINFORCE (Monte-Carlo Policy Gradient)
- **Policy representation:** Shared neural network per agent or centralized training with decentralized execution (CTDE)
- **Objective:** $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$
- **Baseline:** State-value function $V(s)$ to reduce variance

#### 2.1.2 Proximal Policy Optimization (PPO)
- **Objective:** Clipped surrogate objective
  $$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$
- **Advantage estimation:** Generalized Advantage Estimation (GAE) with $\lambda = 0.95$
- **Update:** Multiple epochs of mini-batch SGD on collected rollouts

### 2.2 Architecture
- **Actor network:** MLP `[obs_dim → 64 → 64 → action_dim]` with Tanh output for continuous actions
- **Critic network:** MLP `[global_state_dim → 64 → 64 → 1]` (for centralized critic if using CTDE)
- **Shared parameters:** All agents share the same policy network parameters

### 2.3 Why These Methods?
- REINFORCE serves as a foundational policy-gradient baseline
- PPO is the current standard for stable policy-gradient updates in continuous control

---

## 3. Experiment Settings

### 3.1 Environment Configuration
| Parameter | Value |
|-----------|-------|
| Number of agents | 3 |
| Number of landmarks | 3 |
| Episode length | 25 timesteps |
| World size | 2.0 × 2.0 |
| Agent size | 0.15 |
| Discrete action space option | **Off** (continuous control) |

### 3.2 Hyperparameters
| Hyperparameter | REINFORCE | PPO |
|----------------|-----------|-----|
| Learning rate (actor) | 3e-4 | 3e-4 |
| Learning rate (critic) | 3e-4 | 3e-4 |
| Discount factor $\gamma$ | 0.95 | 0.99 |
| GAE $\lambda$ | N/A | 0.95 |
| PPO clip $\epsilon$ | N/A | 0.2 |
| PPO epochs per update | N/A | 10 |
| Batch size (episodes) | 1 | 32 |
| Hidden layers | [64, 64] | [64, 64] |
| Optimizer | Adam | Adam |
| Total training episodes | 5000 | 5000 |

### 3.3 Evaluation Protocol
- **Random seeds:** 3 independent runs (seeds 0, 1, 2)
- **Evaluation frequency:** Every 100 training episodes, run 10 evaluation episodes without exploration noise
- **Metrics:**
  1. Average episode return
  2. Minimum agent-landmark distance (coverage quality)
  3. Collision rate between agents

### 3.4 Hardware
- CPU / GPU: [To be filled]

---

## 4. Results

> *[Placeholder for learning curves, tables, and figures]*

### 4.1 Learning Curves
- Plot: Average episode return vs. training episodes (mean ± std over 3 seeds)
- Separate subplots for REINFORCE and PPO

### 4.2 Performance Comparison
| Algorithm | Final Avg. Return | Coverage Distance ↓ | Collision Rate ↓ |
|-----------|-------------------|---------------------|------------------|
| REINFORCE | [To be filled] | [To be filled] | [To be filled] |
| PPO | [To be filled] | [To be filled] | [To be filled] |

### 4.3 Qualitative Analysis
- Trajectory visualizations of trained agents
- Failure case analysis (e.g., two agents chasing the same landmark)

---

## 5. Findings & Discussion

### 5.1 Key Observations
- *[To be filled after experiments]*

### 5.2 Challenges & Failure Modes
- Non-stationarity due to simultaneously learning agents
- Sparse / shaped reward trade-offs
- Credit assignment in cooperative multi-agent setting

### 5.3 Limitations
- Limited to small-scale scenario (3 agents, 3 landmarks)
- No communication mechanism between agents
- REINFORCE's high variance in multi-agent setting

### 5.4 Future Work
- Implement MAPPO (Multi-Agent PPO) with centralized value function
- Introduce inter-agent communication channels
- Scale to more agents and dynamic landmark configurations

---

## 6. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
2. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229–256.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
4. Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. *NeurIPS 2017*.
5. Mordatch, I., & Abbeel, P. (2017). Emergence of Grounded Compositional Language in Multi-Agent Populations. *arXiv:1703.04908*.

---

## Appendix

### A.1 Repository Structure
```
marl-ppo-llm-simple-spread/
├── README.md
├── requirements.txt
├── PROJECT_REPORT_DRAFT.md
├── SUBMISSION_CHECKLIST.md
├── src/
│   ├── train_reinforce.py
│   ├── train_ppo.py
│   ├── models.py
│   └── utils.py
├── configs/
│   ├── reinforce_config.yaml
│   └── ppo_config.yaml
└── results/
    ├── learning_curves.png
    └── trajectories/
```

### A.2 Running Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Train REINFORCE
python src/train_reinforce.py --config configs/reinforce_config.yaml

# Train PPO
python src/train_ppo.py --config configs/ppo_config.yaml
```
