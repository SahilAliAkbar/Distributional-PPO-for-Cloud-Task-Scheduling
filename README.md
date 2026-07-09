

## Key Features

* Distributional Reinforcement Learning using Quantile Regression
* Attention-based Actor Network for VM selection
* Quantile Critic trained using Quantile Huber Loss
* CVaR-based risk-sensitive scheduling
* Energy-aware reward design
* VM consolidation strategy to reduce power consumption
* Optimization of both energy consumption and makespan
* Energy-Makespan Ratio (EM Ratio) based evaluation
* Support for heterogeneous VM environments
* Evaluation against heuristic and DRL baselines

---

## Reward Function

The reward function jointly optimizes energy consumption and task completion time:

```text
Reward =
    Energy Cost
  + Makespan Cost
  + Consolidation Bonus
```

Specifically:

```text
energy_cost = -(energy_weight * power * runtime / 1000)

makespan_cost = -(makespan_weight * runtime / 50)

consolidation_bonus = 0.5 * vm_utilization
```

The objective is to:

* Minimize energy consumption
* Minimize task runtime
* Encourage VM consolidation
* Improve overall scheduling efficiency

---

## Evaluation Metrics

The framework evaluates schedulers using three metrics:

* Total Energy Consumption (Joules)
* Makespan
* Energy-Makespan Ratio (EM Ratio)

The EM Ratio is defined as:

```text
EM Ratio = Total Energy Consumption / Makespan
```

A lower EM Ratio indicates a better trade-off between energy efficiency and execution speed.

---

## Algorithms Compared

### Classical Scheduling Algorithms

* FCFS
* Round Robin
* Priority Scheduling
* Random Scheduling

### Reinforcement Learning Baselines

* DQN
* DDQN
* Standard PPO
* Distributional PPO (Proposed)

---

## Experimental Results

| Algorithm                     |     Energy (J) |     Makespan |  EM Ratio |
| ----------------------------- | -------------: | -----------: | --------: |
| FCFS                          |   1,847,240.59 |    18,041.30 |    101.29 |
| Round Robin                   |   1,454,313.71 |     4,691.33 |    315.01 |
| Priority Scheduling           |   1,193,938.09 |     4,766.47 |    268.36 |
| Random                        |   1,480,130.71 |     6,574.54 |    229.22 |
| DQN                           |   1,330,598.17 |    11,647.82 |    117.23 |
| DDQN                          |   1,718,404.21 |    15,767.61 |    107.00 |
| Standard PPO                  |   1,430,424.69 |    14,144.95 |     98.76 |
| **Distributional PPO (Ours)** | **870,356.55** | **9,089.42** | **95.11** |
<img width="1500" height="1200" alt="pareto" src="https://github.com/user-attachments/assets/9d6fe9bb-acb8-40b2-a179-8dd1f65a015b" />
<img width="2850" height="750" alt="comparison" src="https://github.com/user-attachments/assets/61fcf7f2-0239-4a24-961e-3e7ffc3204f1" />
<img width="2850" height="750" alt="learning_curves" src="https://github.com/user-attachments/assets/81a69888-1450-4963-b75b-b225801ac698" />
### Key Observations

* Distributional PPO achieves the **lowest energy consumption** among all evaluated methods.
* Distributional PPO also achieves the **best Energy-Makespan Ratio**, demonstrating the best balance between energy efficiency and execution speed.
* Round Robin and Priority Scheduling achieve very low makespan values but incur significantly higher EM Ratios due to inefficient energy usage.
* Standard PPO improves over DQN and DDQN but remains less energy efficient than the proposed Distributional PPO approach.
* The attention-based distributional critic enables more robust scheduling decisions under uncertain workloads.

---

## Summary

The proposed Distributional PPO scheduler successfully balances two conflicting objectives:

* Reducing total energy consumption
* Maintaining competitive execution performance

By modelling the full return distribution rather than a scalar expected value, the scheduler learns policies that achieve superior long-term energy efficiency while maintaining acceptable makespan performance.
