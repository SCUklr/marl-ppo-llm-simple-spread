# IERG 5350 Course Project Technical Report — Draft

> **Selected Topic:** #5 — Multi-Agent Reinforcement Learning with Independent PPO and LLM-assisted Guidance  
> **Environment:** Multi-Agent Particle Environment (MPE) – Simple Spread  
> **Team:** [Your Names]  
> **Date:** [Date]

---

## 1. Introduction & Problem Setting

### 1.1 Motivation
Multi-agent coordination is a fundamental challenge in reinforcement learning. In this project, we study **PettingZoo Simple Spread**, a cooperative Multi-Agent Particle Environment (MPE) task where several agents must cover multiple landmarks while avoiding collisions. We train **Independent PPO (IPPO)** agents, compare them with a basic **Multi-Agent PPO / MAPPO-style** method, and investigate whether lightweight **LLM-assisted high-level guidance** can improve coordination, credit assignment, and collaboration quality.

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
- Can Independent PPO learn effective cooperative policies in the Simple Spread environment?
- Does a MAPPO-style centralized critic improve sample efficiency or final collaboration performance compared with IPPO?
- Can LLM-assisted guidance improve landmark coverage, reduce collisions, or produce more stable cooperation patterns?
- What are the major failure modes, such as credit assignment errors, agent collision, and landmark coverage imbalance?

---

## 2. Approach / Methodology

### 2.1 Algorithms

#### 2.1.1 Independent PPO (IPPO)
- **Policy representation:** Each agent is trained with PPO using its own local observation. Agents may share policy parameters for sample efficiency while still executing decentralized actions.
- **Objective:** Standard PPO clipped surrogate objective applied independently to each agent's rollout data.
- **Training signal:** Local observations and environment rewards from Simple Spread, with evaluation based on team-level cooperation metrics.

#### 2.1.2 Basic Multi-Agent PPO / MAPPO-Style Baseline
- **Policy representation:** Decentralized actors with a centralized value function that can access global state or concatenated agent observations during training.
- **Objective:** PPO clipped surrogate objective:
  $$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$
- **Advantage estimation:** Generalized Advantage Estimation (GAE) with $\lambda = 0.95$
- **Goal:** Test whether centralized training improves credit assignment and coordination compared with IPPO.

#### 2.1.3 LLM-Assisted Guidance
- **Guidance source:** A lightweight LLM API module generates high-level natural language suggestions at episode reset or fixed intervals.
- **Conversion to RL signal:** Guidance is converted into reward shaping, sub-goal assignment, or strategy labels rather than querying the LLM at every timestep.
- **Design constraint:** The LLM module is optional so that IPPO and MAPPO experiments remain reproducible without an API key.

### 2.2 Architecture
- **Actor network:** MLP `[obs_dim -> 64 -> 64 -> action_dim]` with Tanh output for continuous actions
- **IPPO critic:** Local value network based on each agent's observation
- **MAPPO-style critic:** Centralized value network based on global state or concatenated observations
- **Shared parameters:** Agents may share actor parameters to reduce variance and improve sample efficiency

### 2.3 Why These Methods?
- IPPO is a simple and strong baseline for decentralized multi-agent reinforcement learning.
- MAPPO-style centralized training directly targets credit assignment and non-stationarity in cooperative MARL.
- LLM-assisted guidance adds an interpretable high-level prior without requiring local LLM training or deployment.

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
| Hyperparameter | IPPO | MAPPO-style | LLM-assisted IPPO |
|----------------|------|-------------|-------------------|
| Learning rate (actor) | 3e-4 | 3e-4 | 3e-4 |
| Learning rate (critic) | 3e-4 | 3e-4 | 3e-4 |
| Discount factor $\gamma$ | 0.99 | 0.99 | 0.99 |
| GAE $\lambda$ | 0.95 | 0.95 | 0.95 |
| PPO clip $\epsilon$ | 0.2 | 0.2 | 0.2 |
| PPO epochs per update | 10 | 10 | 10 |
| Batch size (episodes) | 32 | 32 | 32 |
| Hidden layers | [64, 64] | [64, 64] | [64, 64] |
| Optimizer | Adam | Adam | Adam |
| Total training episodes | 5000 | 5000 | 5000 |
| LLM call frequency | N/A | N/A | Episode reset / fixed interval |

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
- Compare IPPO, MAPPO-style PPO, and LLM-assisted IPPO

### 4.2 Performance Comparison
| Algorithm | Final Avg. Return | Coverage Distance ↓ | Collision Rate ↓ | Notes |
|-----------|-------------------|---------------------|------------------|-------|
| IPPO | [To be filled] | [To be filled] | [To be filled] | Decentralized PPO baseline |
| MAPPO-style PPO | [To be filled] | [To be filled] | [To be filled] | Centralized critic baseline |
| LLM-assisted IPPO | [To be filled] | [To be filled] | [To be filled] | High-level guidance extension |

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
- LLM guidance depends on prompt design and should not be queried at every timestep due to latency and cost

### 5.4 Future Work
- Implement a full MAPPO variant with more advanced centralized critics
- Introduce inter-agent communication channels
- Scale to more agents and dynamic landmark configurations

---

## 6. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
3. Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. *NeurIPS 2022*.
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
├── configs/
│   ├── ippo.yaml
│   ├── mappo.yaml
│   └── llm_guidance.yaml
└── results/
    ├── learning_curves.png
    └── trajectories/
```

### A.2 Running Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Train IPPO
python src/train.py --config configs/ippo.yaml

# Train MAPPO-style PPO
python src/train.py --config configs/mappo.yaml

# Train LLM-assisted IPPO
python src/train.py --config configs/llm_guidance.yaml
```
