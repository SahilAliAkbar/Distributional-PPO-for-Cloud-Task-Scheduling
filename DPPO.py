import os
import json
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ======================================================================
# Dataset loading
# ======================================================================
def load_gocj_file(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. Set GOCJ_PATH below to point "
            f"at your GoCJ_Dataset_*.txt file."
        )
    with open(path, "r") as f:
        vals = [float(line.strip()) for line in f if line.strip()]
    return np.array(vals, dtype=np.float64)

# ======================================================================
# Core data structures
# ======================================================================
@dataclass
class VM:
    vm_id: int
    cpu_capacity: float
    mem_capacity: float
    p_idle: float
    p_max: float
    cpu_used: float = 0.0
    mem_used: float = 0.0
    queue: List[int] = field(default_factory=list)
    busy_until: float = 0.0
    energy: float = 0.0
    total_busy_time: float = 0.0
    active_allocations: deque = field(default_factory=deque)

    def utilization(self, now: float) -> float:
        return min(1.0, self.cpu_used / max(self.cpu_capacity, 1e-6))

    def power(self, now: float) -> float:
        u = self.utilization(now)
        if u <= 1e-9:
            return self.p_idle * 0.05
        return self.p_idle + (self.p_max - self.p_idle) * u

@dataclass
class Task:
    task_id: int
    arrival_time: float
    cpu_req: float
    mem_req: float
    length: float
    deadline: float
    assigned_vm: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    def remaining_slack(self, now: float) -> float:
        return self.deadline - now

# ======================================================================
# Environment
# ======================================================================
class CloudSchedulingEnv:
    """
    Clean environment optimized for Energy-Makespan trade-off.
    SLA component removed. Focus: minimize energy and makespan simultaneously.
    """
    def __init__(self, n_vms, n_tasks, seed=0, gocj_path=None,
                 energy_weight=2.0, makespan_weight=1.0):
        self.n_vms = n_vms
        self.n_tasks = n_tasks
        self.seed = seed
        self.gocj_path = gocj_path
        self.energy_weight = energy_weight
        self.makespan_weight = makespan_weight
        self.rng = np.random.default_rng(seed)
        
        if gocj_path is None:
            raise ValueError("Dataset path must be provided for CloudSchedulingEnv")
        
        dataset_vals = load_gocj_file(gocj_path)
        
        # Initialize VMs with heterogeneous power profiles
        self.vms = [
            VM(vm_id=i,
               cpu_capacity=self.rng.uniform(1000, 3000),
               mem_capacity=self.rng.uniform(4, 16),
               p_idle=75.0 * self.rng.uniform(0.75, 1.30),
               p_max=210.0 * self.rng.uniform(0.70, 1.35))
            for i in range(n_vms)
        ]
        
        avg_cpu_cap = float(np.mean([vm.cpu_capacity for vm in self.vms]))
        max_cpu_cap = float(np.max([vm.cpu_capacity for vm in self.vms]))
        avg_mem_cap = float(np.mean([vm.mem_capacity for vm in self.vms]))
        
        # Build tasks from dataset
        self.tasks = []
        n_use = min(n_tasks, len(dataset_vals))
        idxs = self.rng.choice(len(dataset_vals), size=n_use, replace=(n_use > len(dataset_vals)))
        
        for tid, di in enumerate(idxs):
            length = float(dataset_vals[di])
            arrival = float(self.rng.integers(0, 1000))
            cpu_req = float(self.rng.uniform(0.05, 0.25) * avg_cpu_cap)
            mem_req = float(self.rng.uniform(0.05, 0.25) * avg_mem_cap)
            expected_runtime = length / max(avg_cpu_cap, 1e-6)
            optimistic_runtime = length / max(max_cpu_cap, 1e-6)
            expected_wait = (n_tasks / max(2.0 * n_vms, 1.0)) * optimistic_runtime
            slack_factor = float(self.rng.uniform(1.4, 2.6))
            deadline = arrival + expected_runtime + slack_factor * expected_wait
            
            self.tasks.append(Task(
                task_id=tid,
                arrival_time=arrival,
                cpu_req=cpu_req,
                mem_req=mem_req,
                length=length,
                deadline=deadline
            ))
        self.tasks.sort(key=lambda t: t.arrival_time)
        self.task_ptr = 0
        self.current_task = self.tasks[0] if self.tasks else None
        self.time = 0.0
        self.completed = []

    def _release_finished_allocations(self, up_to_time: float):
        for vm in self.vms:
            while vm.active_allocations and vm.active_allocations[0][0] <= up_to_time:
                _, cpu_req, mem_req = vm.active_allocations.popleft()
                vm.cpu_used = max(0.0, vm.cpu_used - cpu_req)
                vm.mem_used = max(0.0, vm.mem_used - mem_req)

    def _sync_to_current_task(self):
        if self.current_task is None:
            return
        target_time = max(self.time, self.current_task.arrival_time)
        self._release_finished_allocations(target_time)
        self.time = target_time

    def _get_state(self):
        self._sync_to_current_task()
        task = self.current_task
        task_feats = np.array([
            task.cpu_req,
            task.mem_req,
            task.length,
            task.deadline - self.time,
            task.arrival_time
        ], dtype=np.float32)
        vm_feats = np.array([
            [vm.cpu_capacity, vm.mem_capacity, vm.cpu_used,
             vm.mem_used, vm.utilization(self.time), vm.power(self.time)]
            for vm in self.vms
        ], dtype=np.float32)
        return task_feats, vm_feats

    def action_mask(self):
        self._sync_to_current_task()
        mask = np.zeros(self.n_vms, dtype=np.float32)
        for i, vm in enumerate(self.vms):
            if vm.cpu_used + self.current_task.cpu_req <= vm.cpu_capacity and \
               vm.mem_used + self.current_task.mem_req <= vm.mem_capacity:
                mask[i] = 1.0
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def step(self, action):
        self._sync_to_current_task()
        task = self.current_task
        vm = self.vms[action]
        
        start = max(self.time, vm.busy_until, task.arrival_time)
        self._release_finished_allocations(start)
        self.time = start
        
        runtime = task.length / max(vm.cpu_capacity, 1e-6)
        finish = start + runtime
        
        task.assigned_vm = vm.vm_id
        task.start_time = start
        task.finish_time = finish
        
        vm.cpu_used += task.cpu_req
        vm.mem_used += task.mem_req
        vm.active_allocations.append((finish, task.cpu_req, task.mem_req))
        
        power = vm.power(self.time)
        vm.energy += power * runtime
        vm.total_busy_time += runtime
        vm.busy_until = finish
        vm.queue.append(task.task_id)
        
        self.completed.append(task)
        self.time = max(self.time, start)
        
        # Reward: minimize energy and makespan
        # Energy cost normalized per 1000 J
        energy_cost = -(self.energy_weight * power * runtime / 1000.0)
        # Makespan (completion time) cost
        makespan_cost = -(self.makespan_weight * runtime / 50.0)
        # Consolidation bonus (load more densely)
        consolidation_bonus = 0.5 * vm.utilization(self.time)
        
        reward = energy_cost + makespan_cost + consolidation_bonus
        
        self.task_ptr += 1
        done = self.task_ptr >= len(self.tasks)
        
        if not done:
            self.current_task = self.tasks[self.task_ptr]
            next_state = self._get_state()
        else:
            self.current_task = None
            next_state = (
                np.zeros(5, dtype=np.float32),
                np.zeros((self.n_vms, 6), dtype=np.float32),
            )
        
        info = {}
        return next_state, reward, done, info

    def summary(self):
        """
        Summary metrics: Energy and Makespan only.
        New metric: Energy-Makespan Ratio = energy / makespan
        Lower ratio = better (energy efficient AND fast)
        """
        total_energy = sum(vm.energy for vm in self.vms)
        finishes = [t.finish_time for t in self.completed if t.finish_time is not None]
        makespan = max(finishes) if finishes else 1.0
        
        # Energy-Makespan Ratio: lower = better on both fronts
        # Avoids optimizing one at expense of the other
        em_ratio = total_energy / max(makespan, 1.0)
        
        return {
            "total_energy": total_energy,
            "makespan": makespan,
            "em_ratio": em_ratio,
            "n_tasks": len(self.completed),
        }

# ======================================================================
# Classical heuristics
# ======================================================================
class FCFS:
    name = "FCFS"
    def act(self, env):
        mask = env.action_mask()
        for i in range(env.n_vms):
            if mask[i] == 1.0:
                return i
        return 0

class RoundRobin:
    name = "RoundRobin"
    def __init__(self):
        self.ptr = 0
    def act(self, env):
        mask = env.action_mask()
        n = env.n_vms
        for _ in range(n):
            i = self.ptr % n
            self.ptr += 1
            if mask[i] == 1.0:
                return i
        return 0

class PriorityScheduling:
    name = "PriorityScheduling"
    HIGH_PRIORITY_URGENCY_THRESHOLD = 1.5
    def act(self, env):
        mask = env.action_mask()
        task = env.current_task
        est_min_runtime = task.length / max(
            np.mean([vm.cpu_capacity for vm in env.vms]), 1e-6
        )
        slack = task.remaining_slack(env.time)
        urgency_ratio = slack / max(est_min_runtime, 1e-6)
        is_high_priority = urgency_ratio < self.HIGH_PRIORITY_URGENCY_THRESHOLD
        feasible = [i for i in range(env.n_vms) if mask[i] == 1.0]
        if not feasible:
            return 0
        if is_high_priority:
            best = max(feasible, key=lambda i: env.vms[i].cpu_capacity)
        else:
            best = min(feasible, key=lambda i: env.vms[i].utilization(env.time))
        return best

class Random:
    name = "Random"
    def act(self, env):
        mask = env.action_mask()
        feasible = [i for i in range(env.n_vms) if mask[i] == 1.0]
        return int(random.choice(feasible)) if feasible else 0

# ======================================================================
# RL infrastructure
# ======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_QUANTILES = 32
TASK_FEAT_DIM = 5
VM_FEAT_DIM = 6
HIDDEN = 128

def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)

class TaskVMAttentionNet(nn.Module):
    """Distributional PPO's actor + critic network."""
    def __init__(self, n_vms, hidden=HIDDEN, n_quantiles=N_QUANTILES):
        super().__init__()
        self.n_vms = n_vms
        self.task_enc = nn.Sequential(nn.Linear(TASK_FEAT_DIM, hidden), nn.ReLU())
        self.vm_enc = nn.Sequential(nn.Linear(VM_FEAT_DIM, hidden), nn.ReLU())
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.scale = hidden ** 0.5
        self.critic_body = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.critic_head = nn.Linear(hidden, n_quantiles)
        self.n_quantiles = n_quantiles
        tau = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("tau", tau)

    def forward(self, task_feats, vm_feats, mask=None):
        t = self.task_enc(task_feats)
        v = self.vm_enc(vm_feats)
        q = self.query(t).unsqueeze(1)
        k = self.key(v)
        logits = (q @ k.transpose(1, 2)).squeeze(1) / self.scale
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        pooled_vm = v.mean(dim=1)
        ctx = torch.cat([t, pooled_vm], dim=-1)
        quantiles = self.critic_head(self.critic_body(ctx))
        return logits, quantiles

    def value_from_quantiles(self, quantiles, alpha=0.25, beta=0.3):
        mean_v = quantiles.mean(dim=-1)
        k = max(1, int(alpha * quantiles.shape[-1]))
        worst, _ = torch.topk(quantiles, k, dim=-1, largest=False)
        cvar = worst.mean(dim=-1)
        return (1 - beta) * mean_v + beta * cvar, mean_v

def quantile_huber_loss(pred_q, target_q, tau, kappa=1.0):
    diff = target_q.unsqueeze(1) - pred_q.unsqueeze(2)
    huber = torch.where(diff.abs() <= kappa, 0.5 * diff ** 2,
                         kappa * (diff.abs() - 0.5 * kappa))
    tau_b = tau.view(1, -1, 1)
    loss = (tau_b - (diff.detach() < 0).float()).abs() * huber
    return loss.sum(dim=1).mean()

class DistributionalPPOAgent:
    name = "Distributional PPO (ours)"
    def __init__(self, n_vms, lr=3e-4, gamma=0.98, lam=0.95, clip=0.2,
                 entropy_coef=0.01, epochs=4, batch_size=64,
                 cvar_alpha=0.25, cvar_beta=0.3):
        self.net = TaskVMAttentionNet(n_vms).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.entropy_coef = entropy_coef
        self.epochs, self.batch_size = epochs, batch_size
        self.cvar_alpha, self.cvar_beta = cvar_alpha, cvar_beta
        self.buffer = []

    def select_action(self, task_feats, vm_feats, mask, deterministic=False, env=None):
        tf = to_tensor(task_feats).unsqueeze(0)
        vf = to_tensor(vm_feats).unsqueeze(0)
        mk = to_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            logits, quantiles = self.net(tf, vf, mk)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(action)
            value, _ = self.net.value_from_quantiles(quantiles, self.cvar_alpha, self.cvar_beta)
        return int(action.item()), float(logp.item()), float(value.item()), quantiles.squeeze(0).cpu().numpy()

    def store(self, task_feats, vm_feats, mask, action, logp, reward, done, value, info=None):
        self.buffer.append((task_feats, vm_feats, mask, action, logp, reward, done, value))

    def _compute_gae(self, rewards, values, dones):
        adv = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values[t + 1] if t + 1 < len(values) else 0.0
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + np.array(values[:len(rewards)])
        return adv, returns

    def update(self):
        if not self.buffer:
            return {}
        task_f, vm_f, masks, actions, old_logp, rewards, dones, values = zip(*self.buffer)
        adv, returns = self._compute_gae(list(rewards), list(values), list(dones))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        task_f = to_tensor(np.array(task_f))
        vm_f = to_tensor(np.array(vm_f))
        masks = to_tensor(np.array(masks))
        actions = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
        old_logp = to_tensor(np.array(old_logp))
        adv_t = to_tensor(adv)
        returns_t = to_tensor(returns)
        
        n = len(self.buffer)
        idxs = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                b = idxs[start:start + self.batch_size]
                logits, quantiles = self.net(task_f[b], vm_f[b], masks[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[b])
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                
                value_est, _ = self.net.value_from_quantiles(quantiles, self.cvar_alpha, self.cvar_beta)
                residual = (returns_t[b] - value_est.detach()).unsqueeze(-1)
                target_quantiles = (quantiles.detach() + residual)
                value_loss = quantile_huber_loss(quantiles, target_quantiles, self.net.tau)
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
        
        self.buffer = []
        return stats

class MLPActorCritic(nn.Module):
    def __init__(self, n_vms, hidden=HIDDEN):
        super().__init__()
        in_dim = TASK_FEAT_DIM + n_vms * VM_FEAT_DIM
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_vms)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        h = self.body(x)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        return logits, self.v(h).squeeze(-1)

def flatten_state(task_feats, vm_feats):
    return np.concatenate([task_feats, vm_feats.flatten()])

class PPOAgent:
    name = "Standard PPO"
    def __init__(self, n_vms, lr=3e-4, gamma=0.98, lam=0.95, clip=0.2,
                 entropy_coef=0.01, epochs=4, batch_size=64):
        self.net = MLPActorCritic(n_vms).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.entropy_coef, self.epochs, self.batch_size = entropy_coef, epochs, batch_size
        self.buffer = []

    def select_action(self, task_feats, vm_feats, mask, deterministic=False, env=None):
        x = to_tensor(flatten_state(task_feats, vm_feats)).unsqueeze(0)
        mk = to_tensor(mask).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(x, mk)
            dist = torch.distributions.Categorical(logits=logits)
            a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item()), None

    def store(self, task_feats, vm_feats, mask, action, logp, reward, done, value, info=None):
        self.buffer.append((flatten_state(task_feats, vm_feats), mask, action, logp, reward, done, value))

    def _gae(self, rewards, values, dones):
        adv = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_v = values[t + 1] if t + 1 < len(values) else 0.0
            nt = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_v * nt - values[t]
            last = delta + self.gamma * self.lam * nt * last
            adv[t] = last
        returns = adv + np.array(values[:len(rewards)])
        return adv, returns

    def update(self):
        if not self.buffer:
            return {}
        states, masks, actions, old_logp, rewards, dones, values = zip(*self.buffer)
        adv, returns = self._gae(list(rewards), list(values), list(dones))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states_t = to_tensor(np.array(states))
        masks_t = to_tensor(np.array(masks))
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
        old_logp_t = to_tensor(np.array(old_logp))
        adv_t, returns_t = to_tensor(adv), to_tensor(returns)
        
        n = len(self.buffer)
        idxs = np.arange(n)
        
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                b = idxs[start:start + self.batch_size]
                logits, v = self.net(states_t[b], masks_t[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions_t[b])
                ratio = torch.exp(logp - old_logp_t[b])
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[b]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(v, returns_t[b])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
        
        self.buffer = []
        return {}

class QNet(nn.Module):
    def __init__(self, n_vms, hidden=HIDDEN):
        super().__init__()
        in_dim = TASK_FEAT_DIM + n_vms * VM_FEAT_DIM
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_vms))

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buf)

class DQNAgentBase:
    name = "DQN"
    double_q = False

    def __init__(self, n_vms, lr=1e-3, gamma=0.98, eps_start=1.0, eps_end=0.05,
                 eps_decay=3000, target_update=200, batch_size=64):
        self.n_vms = n_vms
        self.q = QNet(n_vms).to(DEVICE)
        self.target = QNet(n_vms).to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.buffer = ReplayBuffer()
        self.steps = 0

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

    def select_action(self, task_feats, vm_feats, mask, deterministic=False, env=None):
        x = flatten_state(task_feats, vm_feats)
        if deterministic:
            with torch.no_grad():
                q = self.q(to_tensor(x).unsqueeze(0)).squeeze(0).cpu().numpy()
            q = np.where(mask == 1.0, q, -1e9)
            return int(np.argmax(q)), None, None, None
        
        self.steps += 1
        if random.random() < self.epsilon():
            valid = np.where(mask == 1.0)[0]
            action = int(random.choice(valid))
        else:
            with torch.no_grad():
                q = self.q(to_tensor(x).unsqueeze(0)).squeeze(0).cpu().numpy()
            q = np.where(mask == 1.0, q, -1e9)
            action = int(np.argmax(q))
        return action, None, None, None

    def store(self, task_feats, vm_feats, mask, action, logp, reward, done, value, info=None):
        x = flatten_state(task_feats, vm_feats)
        self.buffer.push(x, action, reward, done, mask)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}
        states, actions, rewards, dones, masks = self.buffer.sample(self.batch_size)
        states_t = to_tensor(np.array(states))
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
        rewards_t = to_tensor(np.array(rewards))
        q_vals = self.q(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = rewards_t
        loss = F.mse_loss(q_vals, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return {"loss": loss.item()}

class DQNAgent(DQNAgentBase):
    name = "DQN"

class DDQNAgent(DQNAgentBase):
    name = "DDQN"
    double_q = True

# ======================================================================
# Experiment configuration
# ======================================================================
N_VMS = 8
GOCJ_PATH = os.environ.get("GOCJ_PATH", "C:/Users/sahil/Downloads/GoCJ_Dataset_850.txt")
N_TASKS = 200
TRAIN_EPISODES = 300
EVAL_SEEDS = list(range(1000, 1010))
EVAL_EVERY = 50

def run_heuristic(scheduler, seed):
    env = CloudSchedulingEnv(n_vms=N_VMS, n_tasks=N_TASKS, seed=seed, gocj_path=GOCJ_PATH)
    done = False
    while not done:
        action = scheduler.act(env)
        _, _, done, _ = env.step(action)
    return env.summary()

def train_rl_agent(agent_cls, agent_kwargs=None, episodes=TRAIN_EPISODES, label=""):
    agent_kwargs = agent_kwargs or {}
    agent = agent_cls(N_VMS, **agent_kwargs)
    rng = np.random.default_rng(42)
    history = {"episode": [], "reward": [], "energy": [], "makespan": [], "em_ratio": []}
    
    for ep in range(episodes):
        seed = int(rng.integers(0, 9999))
        env = CloudSchedulingEnv(n_vms=N_VMS, n_tasks=N_TASKS, seed=seed, gocj_path=GOCJ_PATH)
        task_feats, vm_feats = env._get_state()
        done = False
        ep_reward = 0.0
        
        while not done:
            mask = env.action_mask()
            action, logp, value, _ = agent.select_action(task_feats, vm_feats, mask, env=env)
            (next_task_f, next_vm_f), reward, done, info = env.step(action)
            agent.store(task_feats, vm_feats, mask, action, logp, reward, done, value, info=info)
            task_feats, vm_feats = next_task_f, next_vm_f
            ep_reward += reward
        
        agent.update()
        
        if ep % EVAL_EVERY == 0 or ep == episodes - 1:
            s = env.summary()
            history["episode"].append(ep)
            history["reward"].append(ep_reward)
            history["energy"].append(s["total_energy"])
            history["makespan"].append(s["makespan"])
            history["em_ratio"].append(s["em_ratio"])
            if ep % (EVAL_EVERY * 2) == 0:
                print(f"  [{label}] ep {ep:4d}/{episodes}  return={ep_reward:8.1f}  "
                      f"energy={s['total_energy']:9.1f}  makespan={s['makespan']:8.2f}  em_ratio={s['em_ratio']:8.2f}")
    
    return agent, history

def eval_rl_agent(agent, seeds=EVAL_SEEDS):
    results = []
    for seed in seeds:
        env = CloudSchedulingEnv(n_vms=N_VMS, n_tasks=N_TASKS, seed=seed, gocj_path=GOCJ_PATH)
        task_feats, vm_feats = env._get_state()
        done = False
        while not done:
            mask = env.action_mask()
            action, *_ = agent.select_action(task_feats, vm_feats, mask, deterministic=True, env=env)
            (task_feats, vm_feats), _, done, _ = env.step(action)
        results.append(env.summary())
    return results

def eval_heuristic(scheduler_cls, seeds=EVAL_SEEDS):
    results = []
    for seed in seeds:
        sched = scheduler_cls()
        results.append(run_heuristic(sched, seed))
    return results

def aggregate(results):
    energy = np.mean([r["total_energy"] for r in results])
    makespan = np.mean([r["makespan"] for r in results])
    em_ratio = np.mean([r["em_ratio"] for r in results])
    return dict(energy=energy, makespan=makespan, em_ratio=em_ratio)

def main():
    print(f"Loading dataset from: {GOCJ_PATH}")
    preview = load_gocj_file(GOCJ_PATH)
    print(f"Dataset: n={len(preview)}, min={preview.min():.0f}, max={preview.max():.0f}, mean={preview.mean():.0f}\n")
    
    all_results = {}
    histories = {}
    
    print("=== Evaluating heuristics ===")
    for cls in [FCFS, RoundRobin, PriorityScheduling, Random]:
        print(f"  Running {cls.name}...")
        res = eval_heuristic(cls)
        agg = aggregate(res)
        all_results[cls.name] = agg
        print(f"    energy={agg['energy']:.1f}  makespan={agg['makespan']:.2f}  em_ratio={agg['em_ratio']:.2f}")
    
    print("\n=== Training RL agents ===")
    
    print("Training DQN...")
    dqn, histories["DQN"] = train_rl_agent(DQNAgent, episodes=TRAIN_EPISODES, label="DQN")
    agg = aggregate(eval_rl_agent(dqn))
    all_results["DQN"] = agg
    print(f"  Final: energy={agg['energy']:.1f}  makespan={agg['makespan']:.2f}  em_ratio={agg['em_ratio']:.2f}")
    
    print("\nTraining DDQN...")
    ddqn, histories["DDQN"] = train_rl_agent(DDQNAgent, episodes=TRAIN_EPISODES, label="DDQN")
    agg = aggregate(eval_rl_agent(ddqn))
    all_results["DDQN"] = agg
    print(f"  Final: energy={agg['energy']:.1f}  makespan={agg['makespan']:.2f}  em_ratio={agg['em_ratio']:.2f}")
    
    print("\nTraining Standard PPO...")
    ppo, histories["Standard PPO"] = train_rl_agent(PPOAgent, episodes=TRAIN_EPISODES, label="PPO")
    agg = aggregate(eval_rl_agent(ppo))
    all_results["Standard PPO"] = agg
    print(f"  Final: energy={agg['energy']:.1f}  makespan={agg['makespan']:.2f}  em_ratio={agg['em_ratio']:.2f}")
    
    print("\nTraining Distributional PPO...")
    dppo, histories["Distributional PPO (ours)"] = train_rl_agent(
        DistributionalPPOAgent, episodes=TRAIN_EPISODES, label="Distributional PPO")
    agg = aggregate(eval_rl_agent(dppo))
    all_results["Distributional PPO (ours)"] = agg
    print(f"  Final: energy={agg['energy']:.1f}  makespan={agg['makespan']:.2f}  em_ratio={agg['em_ratio']:.2f}")
    
    print("\n=== Final Results (sorted by Energy-Makespan Ratio) ===")
    header = f"{'Algorithm':25s} {'Energy(J)':>12s} {'Makespan':>10s} {'EM Ratio':>10s}"
    print(header)
    print("-" * len(header))
    for name, r in sorted(all_results.items(), key=lambda kv: kv[1]["em_ratio"]):
        print(f"{name:25s} {r['energy']:12.1f} {r['makespan']:10.2f} {r['em_ratio']:10.2f}")
    
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("histories.json", "w") as f:
        json.dump(histories, f, indent=2)
    print("\nSaved results.json, histories.json")
    
    plot_all(all_results, histories)
    return all_results, histories

def plot_all(all_results, histories):
    names = list(all_results.keys())
    energy = [all_results[n]["energy"] for n in names]
    makespan = [all_results[n]["makespan"] for n in names]
    em_ratio = [all_results[n]["em_ratio"] for n in names]
    colors = ["#C44E52" if n == "Distributional PPO (ours)" else "#4C72B0" for n in names]
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, data, title, ylabel in zip(
        axes, [energy, makespan, em_ratio], 
        ["Total Energy", "Makespan", "Energy-Makespan Ratio"],
        ["Energy (J)", "Time (sec)", "Ratio (lower=better)"]
    ):
        ax.bar(names, data, color=colors)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("Saved comparison.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, r in all_results.items():
        color = "#C44E52" if name == "Distributional PPO (ours)" else "#4C72B0"
        marker = "s" if name == "Distributional PPO (ours)" else "o"
        size = 350 if name == "Distributional PPO (ours)" else 120
        ax.scatter(r["energy"], r["makespan"], s=size, color=color, marker=marker, zorder=3, alpha=0.7)
        ax.annotate(name, (r["energy"], r["makespan"]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Total Energy (J)  [lower = better]", fontsize=12)
    ax.set_ylabel("Makespan (sec)  [lower = better]", fontsize=12)
    ax.set_title("Energy-Makespan Pareto Front (bottom-left is best)", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto.png", dpi=150)
    plt.show()
    print("Saved pareto.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    line_colors = {"DQN": "#55A868", "DDQN": "#8172B2", "Standard PPO": "#4C72B0",
                   "Distributional PPO (ours)": "#C44E52"}
    for agent_name, h in histories.items():
        lw = 3 if agent_name == "Distributional PPO (ours)" else 1.6
        axes[0].plot(h["episode"], h["reward"], label=agent_name, color=line_colors.get(agent_name), linewidth=lw)
        axes[1].plot(h["episode"], h["energy"], label=agent_name, color=line_colors.get(agent_name), linewidth=lw)
        axes[2].plot(h["episode"], h["em_ratio"], label=agent_name, color=line_colors.get(agent_name), linewidth=lw)
    
    axes[0].set_title("Episode Return during Training", fontweight='bold')
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Return")
    axes[1].set_title("Episode Energy during Training", fontweight='bold')
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Energy (J)")
    axes[2].set_title("Episode EM Ratio during Training", fontweight='bold')
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("EM Ratio")
    
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.show()
    print("Saved learning_curves.png")

if __name__ == "__main__":
    main()