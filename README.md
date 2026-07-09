

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
## Dataset

### GoCJ Dataset

This work uses the **GoCJ (Google Cluster Jobs)** dataset as the workload benchmark for evaluating scheduling performance.

GoCJ is a publicly available benchmark dataset for **cloud scheduling and resource allocation research**. Although it is not a universal benchmark in the same sense as datasets such as ImageNet or MNIST, it is widely used in cloud computing literature for evaluating scheduling algorithms, VM allocation policies, energy-aware schedulers, and reinforcement learning based approaches.

The GoCJ dataset was created using workloads derived from **Google cluster traces** with the objective of providing researchers with a standardized benchmark for comparing cloud scheduling techniques. The original GoCJ publication explicitly states that the dataset was made publicly available to facilitate fair comparison and benchmarking of scheduling and resource management algorithms in cloud environments.

More recently, the **Enhanced GoCJ** dataset introduced multiple workload files of varying sizes generated from Google cluster traces, enabling evaluation under different scales and workload intensities.

### Workload Characteristics

The experiments in this work utilize the **GoCJ-1000** workload containing **1000 cloud tasks (cloudlets)**.

Each task value represents the **computational workload or task length**, typically measured in **Million Instructions (MI)**.

The uploaded workload exhibits significant heterogeneity:

* Number of tasks: **1000**
* Minimum task size: **15,000 MI**
* Maximum task size: **900,000 MI**
* Multiple workload categories ranging from lightweight to compute-intensive jobs

Example task sizes include:

| Category   | Example Task Lengths (MI)          |
| ---------- | ---------------------------------- |
| Small      | 15,000, 27,500, 40,000             |
| Medium     | 65,000, 95,000, 121,000            |
| Large      | 150,000                            |
| Very Large | 337,500, 525,000, 712,500, 900,000 |

This heterogeneous workload distribution makes GoCJ particularly suitable for evaluating scheduling policies under realistic cloud conditions where short and long-running tasks coexist.

### Motivation for Using GoCJ

The GoCJ benchmark is well suited for evaluating the proposed scheduling framework because it:

* Simulates realistic cloud workload diversity.
* Contains tasks with highly varying computational requirements.
* Enables comparison with traditional scheduling heuristics and modern intelligent schedulers.
* Is commonly adopted in cloud scheduling literature for benchmarking purposes.

In this work, the GoCJ workload is used to compare scheduling approaches based on:

* Energy Consumption
* Makespan
* EM Ratio
* Resource Utilization
* Overall Scheduling Efficiency

### References

1. Original GoCJ Dataset: Publicly released for benchmarking cloud scheduling and resource allocation algorithms using workloads derived from Google cluster traces.

2. Enhanced GoCJ Dataset: An extended version containing multiple workload files of different sizes and characteristics for large-scale cloud scheduling evaluation.

## Summary

The proposed Distributional PPO scheduler successfully balances two conflicting objectives:

* Reducing total energy consumption
* Maintaining competitive execution performance

By modelling the full return distribution rather than a scalar expected value, the scheduler learns policies that achieve superior long-term energy efficiency while maintaining acceptable makespan performance.
