---

## 🚀 D-PPO Energy-Efficient Cloud Task Scheduler

AI-powered cloud scheduling using Distributional Proximal Policy Optimization (D-PPO) for energy-efficient, SLA-aware, and adaptive resource optimization.

---

## 📌 Overview

This project presents an advanced cloud task scheduling system based on Distributional Reinforcement Learning, specifically D-PPO (Distributional Proximal Policy Optimization).

Traditional scheduling algorithms struggle to balance **performance, energy efficiency, and reliability** in dynamic environments. This project addresses those limitations by modeling scheduling as a **Markov Decision Process (MDP)** and learning optimal policies through interaction with the environment.

---

## ✨ Key Features

* ⚡ **~70% Reduction in Energy Consumption**
* ⏱ **Competitive Makespan Performance**
* ✅ **Near-Zero SLA Violations**
* 🧠 **Distributional Reinforcement Learning (Uncertainty-Aware)**
* 🔄 **Adaptive Scheduling under Dynamic Workloads**
* 📈 **Stable and Fast Convergence**

---

## 🧠 Methodology

### 🔹 Problem Formulation

* Modeled as a **Markov Decision Process (MDP)**
* States include:

  * Task characteristics
  * VM resource utilization
  * Queue status
* Actions:

  * Assign task → virtual machine
* Reward:

  * Multi-objective (energy, delay, SLA)

---

### 🔹 Core Algorithm

**D-PPO (Distributional Proximal Policy Optimization)**

* Based on:

  * PPO (stable policy updates)
  * Distributional RL (models full return distribution)
* Benefits:

  * Better uncertainty handling
  * Improved decision-making
  * Stable training

---

## 📊 Results

### 🔹 Performance Summary

| Metric             | Improvement       |
| ------------------ | ----------------- |
| Energy Consumption | 🔻 ~70% reduction |
| Makespan           | ⚖️ Competitive    |
| SLA Violations     | ✅ Near-zero       |

---

### 🔹 Visual Results

#### Training Convergence

* Reward stabilizes after ~150–200 episodes
* Low variance → stable policy

#### Algorithm Comparison

* Compared with:

  * FCFS
  * SJF
  * RR
  * EDF
  * Min-Min
  * Max-Min

📌 Add your plots here:

```
/results/training_curve.png
/results/comparison_chart.png
```

---


## ⚙️ Tech Stack

* 🐍 Python
* 📊 NumPy, Pandas
* 📈 Matplotlib
* 🤖 PyTorch / TensorFlow (whichever you used)

---

## 🧪 Experimental Setup

* Custom Python-based simulation
* Multiple evaluation runs (averaged results)
* Metrics:

  * Makespan
  * Energy Consumption
  * SLA Violations

---

## 🏆 Achievements

* 🥇 **Nominated for Best Innovative Project**
* 📊 Demonstrated **significant energy efficiency improvements**
* 🧠 Introduced **Distributional RL in cloud scheduling**

---

## 🔮 Future Work

* 🌐 Integration with real cloud platforms (AWS, Azure)
* ⚡ Faster training & lightweight models
* 🧠 Attention-based / memory-augmented architectures
* 📊 Multi-objective adaptive optimization

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Reinforcement Learning research community
* PPO & Distributional RL foundational works
* Cloud scheduling research contributions

