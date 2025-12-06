# Meta-Learning for Reinforcement Learning in Building Energy Management

**Type:** Master's Thesis

**Author:** Irina Eftimova

**Supervisor:** Alona Zharova

**1st Examiner:** Stefan Lessmann


![Performance Results](outputs/graphs/performance_comparison.png)

---

## Table of Contents

- [Project Overview](#Project-Overview)
- [Performance Summary](#Performance-Summary)
- [Quick Start](#Quick-Start)
    - [Dependencies](#Install-Dependencies)
    - [Setup](#Prepare-Data)
    - [Train Models](#Train-Models)
    - [Evaluate Models](#Evaluate-Models)
    - [Visualisations](#Generate-Figures-and-Tables)
- [Project structure](#Project-structure)


---

## Project Overview

This thesis focuses on applying Meta Reinforcement Learning across six residential households and compares **Meta-RL** against **Standard RL** and **Rule-Based Control (RBC)** for building energy management. The goal is to train agents that can generalize to unseen buildings and optimize energy consumption while maintaining comfort levels.


---

## Performance Summary

Evaluated on Test Buildings (5-6):

| Metric | RBC | RL | Meta-RL | **Meta-RL vs RL** |
|--------|-----|-------------|---------|-------------------|
| **Carbon Reduction** | -8.8% ❌ | 1.2% ❌ | **7.2%** ✅ | **+6%** |
| **Cost Savings** | -8.9% ❌ | 0.9% ❌ | **8.1%** ✅ | **+7.2%** |
| **Peak Load Reduction** | -5.9% ❌ | 1.5% ❌ | **5.9%** ✅ | **+4.4%** |

The Meta-RL Q Agent was trained on Buildings 1-4, and evaluated on two unseen Buildings (5-6). 
The adaptive capability of Meta-RL significantly enhances home energy management, resulting in superior performance compared to the standard RL approach. Specifically, Meta-RL achieved an average 8.10\% improvement in cost savings, 7.24\% reduction in carbon emissions, and 5.87\% reduction in peak energy demand.


---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare Data 

```bash
python eda.py
```
- This loads the CityLearn dataset and preprocesses 6 buildings (phase 3).

### Train Models

**Train Standard RL:**
```bash
python train_sac.py
```
- Trains on Buildings 1-4
- Output: `sac_model.pt`

**Train Meta-RL:**
```bash
python meta_rl_trainer.py
```
- Trains on Buildings 1-4 using enhanced Reptile ML
- Output: `meta_lstmq_reptile.pt`

### Evaluate Models

```bash
python evaluate_all_metrics.py
```

This will:
- Load all three methods (Meta-RL enhanced + Standard RL + RBC)
- Evaluate on test buildings (5-6) 
- Calculate comprehensive metrics (carbon, cost, peak, ramping, load factor, etc.)
- Save detailed results to `outputs/eval/comprehensive_metrics/all_metrics_real.csv`

### Generate Figures and Tables

**Create Figures:**
```bash
python plots.py
```

**Create Metrics Table:**
```bash
python table.py
```
---

---

## Project Structure

```
Code/´
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
│
├── src:
│   ├── agent_setup.py                       # Task loader for 6 buildings
│   ├── eda.py                               # Data loading (CityLearn dataset)
│   ├── environment_wrapper.py               # CityLearn Environment Wrapper for RL
│   ├── evaluate_all_metrics.py              # Comprehensive evaluation & metrics
│   ├── lstm_q_agent.py                      # LSTM Q-Network agent
│   ├── meta_rl_trainer.py                   # Meta-RL training (Reptile)
│   ├── rbc_agent.py                         # Rule-based control agent
│   ├── recurrent_env_wrapper.py             # Wrapper for Meta-RL
│   ├── reward_utils.py                      # Unified reward functions
│   ├── sac_agent.py                         # SAC agent
│   └── train_sac.py                         # SAC training (RL)
│
├── models:
│   ├── sac_model.pt                          # SAC RL model
│   └── meta_lstmq_reptile.pt                 # Meta-RL model
│
├── visuals:
│   ├── plots.py                             # Figures
│   └── table.py                             # Detailed metrics tables
│
└── outputs:
    ├── phase3/                              # Building data (6 buildings)
    └── graphics/                            # Evaluation results & figures
        └──  performance_comparison.png      # Performance across Models
```
