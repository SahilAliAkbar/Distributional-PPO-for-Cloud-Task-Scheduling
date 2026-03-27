"""
D-PPO Cloud Task Scheduler  --  v5 (Best-of-Both)
===================================================
Core  : v1 proven design (action space, reward, QR-DQN, hyperparams)
Shell : v4 improvements  (seaborn plots saved to PNG, enriched obs, diagnostics)

WHY THIS MERGE:
  v1 achieved Energy=411 (-68% vs SJF), Makespan=7019 (+0.2% vs FCFS), SLA=0
  v4 changed too much at once (joint action space, IQN, new reward, new arch)
  and lost the stable training signal.

  Preserved from v1 (the parts that drove good results):
    - Action space: Discrete(N_VMS=4)  -- simple, direct credit assignment
    - Reward:  W_FINISH=0.5, W_ENERGY=1.5, E_NORM=10.3 (absolute energy)
    - Networks: PolicyNet (256-hid, 3 ResBlocks) + QuantileCriticNet (QR-DQN, 64 quant.)
    - Hyperparams: lr_a=2e-4, lr_c=5e-4, eps_clip=0.15, ent_coef=0.01 (fixed),
                   batch=256, n_epochs=8, cvar_alpha=0.25
    - Scheduler:  CosineAnnealingLR (simple, no warm restart)
    - Curriculum:  80 + ep//5, max 200

  Added from v4 (improvements that don't break stability):
    - Enriched obs: spread + queue_depth globals appended  (obs_dim: 17 -> 19)
    - Seaborn plots saved as high-DPI PNGs to ./outputs/
    - Per-component reward diagnostics (comp_hist)
    - 30-window evaluation (was 25)
    - Precise pass/miss target reporting
"""

import os, sys, subprocess

def _pip(*a):
    """Install a package quietly via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a])

# ── Dependency check (run once; re-run script if prompted) ────────────
_MISSING = []
try:
    import torch; torch.zeros(1)
except Exception:
    _MISSING.append("torch")

for _pkg in ["gymnasium", "seaborn"]:
    try: __import__(_pkg)
    except ImportError: _MISSING.append(_pkg)

if _MISSING:
    print(f"Installing missing packages: {_MISSING}")
    print("  torch  : pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("  others : pip install gymnasium seaborn")
    for p in _MISSING:
        if p == "torch":
            _pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        else:
            _pip(p)
    print("Packages installed. Please re-run the script.")
    sys.exit(0)

import copy, gzip, io, math, warnings
import urllib.request
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: saves PNGs, never opens windows
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
OUT_DIR   = "./outputs"
DATA_PATH = "gocj_data.csv"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cpu"

# ── Plot theme ────────────────────────────────────────────────────────
BG, PAN, RED = "#0F0F1A", "#161628", "#FF4B4B"
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor":   PAN,
    "axes.edgecolor":  "#2A2A4A", "axes.labelcolor": "#CCCCCC",
    "text.color":      "#CCCCCC", "xtick.color":     "#AAAAAA",
    "ytick.color":     "#AAAAAA", "grid.color":      "#2A2A4A",
    "grid.alpha":       0.3,      "legend.facecolor": PAN,
    "legend.edgecolor":"#2A2A4A", "legend.labelcolor":"#CCCCCC",
    "font.family":     "DejaVu Sans", "figure.dpi":   150,
})


# =====================================================================
# 1.  VM FLEET & REWARD CONSTANTS  (v1 values -- proven)
# =====================================================================
VM_TYPES = ["std", "std", "premium", "premium"]
VM_SPEED = {"std": 1.0,  "premium": 1.3}
VM_RATE  = {"std": 0.05, "premium": 0.40}
N_VMS    = 4

# v1 reward weights -- do NOT change, these drove Energy=411, SLA=0
W_FINISH = 0.5    # low: don't sacrifice energy chasing makespan
W_ENERGY = 1.5    # dominant signal: route to std VM when free
W_SLA    = 2.0    # hard deadline penalty
E_NORM   = 10.3   # measured mean per-step energy under FCFS at TI=0.35

# Spread normaliser (for enriched obs only -- not used in reward)
SPREAD_NORM = 120.0


def eff_dur(dur, vt):    return dur / VM_SPEED[vt]
def step_energy(dur, vt): return VM_RATE[vt] * eff_dur(dur, vt)


# =====================================================================
# 2.  DATA  (unchanged from v1/v4)
# =====================================================================
_GCS = ("https://commondatastorage.googleapis.com/"
        "clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz")

def _download_gocj(path, max_rows=15000):
    print("  Downloading GOCJ ...")
    try:
        req = urllib.request.Request(_GCS, headers={"User-Agent":"Python"})
        with urllib.request.urlopen(req, timeout=180) as r: raw = r.read()
        print(f"  {len(raw)//1024//1024} MB downloaded.")
    except Exception as e:
        print(f"  Download failed: {e}"); return False
    rows = []
    with gzip.open(io.BytesIO(raw),"rt",errors="replace") as fh:
        for line in fh:
            c = line.strip().split(",")
            if len(c)<11: continue
            try:
                if int(c[5])!=0: continue
                rows.append({"arrival_time": float(c[0])/1e6,
                              "cpu_req": min(float(c[9] or 0.25)*4,4.),
                              "mem_req": min(float(c[10] or 0.25)*16,16.)})
            except: continue
            if len(rows)>=max_rows: break
    if not rows: return False
    rng=np.random.default_rng(SEED)
    df=pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)
    df["arrival_time"]-=df["arrival_time"].iloc[0]
    n=len(df)
    df["duration"]=rng.gamma(1.5,25.,n).clip(1.)
    df["deadline"]=df["arrival_time"]+df["duration"]*rng.uniform(1.5,2.,n)
    df[["arrival_time","cpu_req","mem_req","duration","deadline"]].to_csv(path,index=False)
    print(f"  Saved {n} jobs -> {path}"); return True

def _rescale(df, target_ti=0.35):
    rng=np.random.default_rng(SEED)
    df=df.sort_values("arrival_time").reset_index(drop=True).copy()
    gaps=np.diff(df["arrival_time"].values,prepend=df["arrival_time"].iloc[0])
    gaps=np.maximum(gaps,np.quantile(gaps[gaps>0],0.1) if (gaps>0).any() else 1.)
    tg=df["duration"].mean()/max(target_ti*N_VMS,1e-6)
    ng=tg*(0.7+0.6*gaps/max(gaps.mean(),1e-9))*rng.uniform(0.9,1.1,len(gaps))
    arr=np.cumsum(ng); arr-=arr[0]
    df["arrival_time"]=arr
    df["deadline"]=arr+df["duration"].values*rng.uniform(1.5,2.,len(df))
    return df

def _synthetic(path):
    rng=np.random.default_rng(SEED); n=5000
    arr=np.sort(rng.exponential(26.8,n).cumsum())
    dur=rng.gamma(1.5,25.,n).clip(1.)
    df=pd.DataFrame({"arrival_time":arr,"cpu_req":rng.uniform(.5,4.,n),
                      "mem_req":rng.uniform(1.,12.,n),"duration":dur,
                      "deadline":arr+dur*rng.uniform(1.5,2.,n)})
    df.to_csv(path,index=False); return df

def _ti(df):
    g=np.diff(df["arrival_time"].values); g=g[g>0]
    return float(df["duration"].mean()/(g.mean() if len(g) else 1.)/N_VMS)

class GOCJDataset:
    COLS={"arrival_time","cpu_req","mem_req","duration","deadline"}
    def __init__(self,path=DATA_PATH):
        if os.path.exists(path):
            df=pd.read_csv(path)
            if self.COLS.issubset(df.columns) and len(df)>100:
                if _ti(df)>0.5:
                    df=_rescale(df,0.35); df.to_csv(path,index=False)
                self.data=df; self.source="Cached CSV"
                print(f"  {len(df)} jobs loaded from {path}"); return
        ok=_download_gocj(path)
        if ok:
            df=pd.read_csv(path)
            if _ti(df)>0.5: df=_rescale(df,0.35); df.to_csv(path,index=False)
            self.data=df; self.source="Real Google Cluster Data 2011"
            print(f"  Using real GOCJ ({len(df)} jobs)"); return
        self.data=_synthetic(path); self.source="Synthetic fallback"
        print(f"  Synthetic data ({len(self.data)} jobs)")

    def window(self,n,rng):
        pool=self.data
        start=int(rng.integers(0,max(1,len(pool)-n)))
        rows=pool.iloc[start:start+n].copy().to_dict("records")
        base=rows[0]["arrival_time"]
        for r in rows: r["arrival_time"]-=base; r["deadline"]-=base
        return rows

    def all_tasks(self): return self.data.copy().to_dict("records")

print("Loading dataset ...")
DS   = GOCJDataset(DATA_PATH)
_tiv = _ti(DS.data)
print(f"  Source : {DS.source}")
print(f"  TI     : {_tiv:.3f}  ({'OK' if _tiv<1. else 'OVERLOADED'})\n")


# =====================================================================
# 3.  ENVIRONMENT
#     v1 action space (Discrete N_VMS=4) + v4 enriched obs
#     (spread + queue_depth appended as 2 extra global features)
#     obs_dim = 5 + 3*N_VMS + 2 = 19
# =====================================================================
class Env(gym.Env):
    metadata = {"render_modes": []}
    # v1 core obs (17) + 2 global context features from v4 = 19
    OBS_DIM = 5 + 3*N_VMS + 2

    def __init__(self, eval_mode=False):
        super().__init__()
        self.eval_mode = eval_mode
        self.observation_space = spaces.Box(0.,1.,(self.OBS_DIM,),np.float32)
        self.action_space      = spaces.Discrete(N_VMS)   # v1: 4 actions
        self._r_components: Dict = {}
        self._reset()

    def _reset(self):
        self.queue: List[Dict] = []
        self.vm_fin = np.zeros(N_VMS, np.float64)
        self.vm_erg = np.zeros(N_VMS, np.float64)
        self.now    = 0.
        self.sla    = 0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed); self._reset()
        return self._obs(), {}

    def load(self, tasks):
        self.queue = copy.deepcopy(tasks)

    def step(self, action: int):
        if not self.queue:
            return self._obs(), 0., True, False, {}

        t   = self.queue.pop(0)
        vi  = int(action)
        arr = float(t["arrival_time"])
        dur = float(t["duration"])
        self.now = arr

        # Stochastic duration noise during training (v1 unchanged)
        if not self.eval_mode:
            dur *= float(np.clip(1.+np.random.normal(0,.05),.9,1.1))

        eds  = [eff_dur(dur,VM_TYPES[v])       for v in range(N_VMS)]
        avl  = [max(arr,float(self.vm_fin[v])) for v in range(N_VMS)]
        fins = [avl[v]+eds[v]                  for v in range(N_VMS)]
        ens  = [step_energy(dur,VM_TYPES[v])   for v in range(N_VMS)]

        f_act = fins[vi]; e_act = ens[vi]
        self.vm_fin[vi]  = f_act
        self.vm_erg[vi] += e_act

        dl  = float(t.get("deadline",float("inf")))
        lat = max(0., f_act - dl)
        self.sla += int(lat>0)

        # ── v1 reward formula (proven) ────────────────────────────────
        bf, wf = min(fins), max(fins)
        r_finish = -W_FINISH * (f_act - bf) / (wf - bf + 1e-6)
        r_energy = -W_ENERGY * e_act / E_NORM
        r_sla    = -W_SLA * min(1., lat / max(eds[vi],1.)) if lat>0 else 0.
        r = r_finish + r_energy + r_sla

        self._r_components = {"finish": r_finish,
                               "energy": r_energy,
                               "sla":    r_sla}

        return self._obs(), r, len(self.queue)==0, False, {}

    def _obs(self) -> np.ndarray:
        T = 800.
        # -- Task features (v1) ----------------------------------------
        if not self.queue:
            to = [0.]*5
        else:
            t  = self.queue[0]
            dl = float(t.get("deadline",float("inf")))
            ttd = (1. if dl==float("inf") or dl<=self.now
                   else float(np.clip((dl-self.now)/T,0,1)))
            to = [np.clip(t["cpu_req"]/4.,0,1),
                  np.clip(t["mem_req"]/16.,0,1),
                  np.clip(t["duration"]/T,0,1),
                  ttd,
                  np.clip(t["arrival_time"]/100000.,0,1)]

        # -- VM features (v1) ------------------------------------------
        vo = []
        for i in range(N_VMS):
            rem = max(0.,float(self.vm_fin[i])-self.now)
            vo += [np.clip(rem/T,0,1),
                   np.clip(self.vm_erg[i]/800.,0,1),
                   float(VM_TYPES[i]=="premium")]

        # -- Global context (v4 addition -- does not disturb reward) ---
        spread = float(self.vm_fin.max()-self.vm_fin.min())
        qdepth = len(self.queue)
        go = [np.clip(spread/SPREAD_NORM,0,1),
              np.clip(qdepth/200.,0,1)]

        return np.array(to+vo+go, dtype=np.float32)

    def mask(self) -> np.ndarray:
        m = np.ones(N_VMS, bool)
        if not self.queue: return m
        t  = self.queue[0]; dl=float(t.get("deadline",float("inf")))
        if dl==float("inf"): return m
        arr,dur=float(t["arrival_time"]),float(t["duration"])
        for i in range(N_VMS):
            m[i]=max(arr,float(self.vm_fin[i]))+eff_dur(dur,VM_TYPES[i])<=dl
        return m if m.any() else np.ones(N_VMS,bool)

    def makespan(self): return float(self.vm_fin.max())
    def energy(self):   return float(self.vm_erg.sum())
    def sla_viol(self): return self.sla

print("Environment ready.\n")


# =====================================================================
# 4.  BASELINES  (v1 unchanged)
# =====================================================================
class _Sched:
    def _put(self,env,task,vi):
        arr,dur=float(task["arrival_time"]),float(task["duration"])
        fin=max(arr,float(env.vm_fin[vi]))+eff_dur(dur,VM_TYPES[vi])
        env.vm_fin[vi]=fin; env.vm_erg[vi]+=step_energy(dur,VM_TYPES[vi])
        if fin>float(task.get("deadline",float("inf"))): env.sla+=1
    def _argmin_fin(self,env,task):
        arr,dur=float(task["arrival_time"]),float(task["duration"])
        return int(np.argmin([max(arr,float(env.vm_fin[v]))+eff_dur(dur,VM_TYPES[v])
                               for v in range(env.action_space.n)]))

class FCFS(_Sched):
    def run(self,env,tasks):
        for t in tasks: self._put(env,t,self._argmin_fin(env,t))

class RoundRobin(_Sched):
    def run(self,env,tasks):
        for i,t in enumerate(tasks): self._put(env,t,i%env.action_space.n)

class SJF(_Sched):
    def run(self,env,tasks):
        for t in sorted(tasks,key=lambda x:x["duration"]):
            self._put(env,t,self._argmin_fin(env,t))

class EDF(_Sched):
    def run(self,env,tasks):
        for t in sorted(tasks,key=lambda x:x.get("deadline",float("inf"))):
            self._put(env,t,self._argmin_fin(env,t))

class MinMin(_Sched):
    def run(self,env,tasks):
        rem=list(tasks)
        while rem:
            bt,bv,bc=0,0,float("inf")
            for ti,t in enumerate(rem):
                arr,dur=float(t["arrival_time"]),float(t["duration"])
                for vi in range(env.action_space.n):
                    ct=max(arr,float(env.vm_fin[vi]))+eff_dur(dur,VM_TYPES[vi])
                    if ct<bc: bc,bt,bv=ct,ti,vi
            self._put(env,rem.pop(bt),bv)

class MaxMin(_Sched):
    def run(self,env,tasks):
        rem=list(tasks)
        while rem:
            best=[(max(float(t["arrival_time"]),float(env.vm_fin[vi]))
                   +eff_dur(float(t["duration"]),VM_TYPES[vi]),vi)
                  for t in rem
                  for vi in [min(range(env.action_space.n),
                                  key=lambda v:max(float(t["arrival_time"]),
                                                    float(env.vm_fin[v]))
                                              +eff_dur(float(t["duration"]),VM_TYPES[v]))]]
            ch=int(np.argmax([b[0] for b in best]))
            self._put(env,rem.pop(ch),best[ch][1])

print("Baselines ready.\n")


# =====================================================================
# 5.  NETWORKS  (v1 architecture -- proven stable)
#     PolicyNet:         Encoder(19->256, 3 ResBlocks) + head(256->4) + urgency
#     QuantileCriticNet: QR-DQN, 64 FIXED quantiles, QR-Huber loss
#     (obs_dim bumped 17->19 to absorb the 2 global context features)
# =====================================================================
def _lin(i, o, gain=np.sqrt(2)):
    l=nn.Linear(i,o)
    nn.init.orthogonal_(l.weight,gain=gain)
    nn.init.constant_(l.bias,0.)
    return l

class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(_lin(d,d),nn.LayerNorm(d),nn.Tanh(),
                                _lin(d,d),nn.LayerNorm(d))
    def forward(self,x): return torch.tanh(self.net(x)+x)

class Encoder(nn.Module):
    """v1: hid=256, 3 ResBlocks."""
    def __init__(self,in_d,hid=256,blocks=3):
        super().__init__()
        self.proj  =nn.Sequential(_lin(in_d,hid),nn.LayerNorm(hid),nn.Tanh())
        self.blocks=nn.Sequential(*[ResBlock(hid) for _ in range(blocks)])
    def forward(self,x): return self.blocks(self.proj(x))

class PolicyNet(nn.Module):
    """v1 actor: flat obs -> logits over N_VMS=4 VMs."""
    def __init__(self,obs_d,act_d,hid=256):
        super().__init__()
        self.enc  = Encoder(obs_d,hid)
        self.head = _lin(hid,act_d,gain=0.01)
        self.urg  = _lin(hid,1,gain=1.0)   # auxiliary urgency head
    def forward(self,x):
        h=self.enc(x)
        return self.head(h), torch.sigmoid(self.urg(h))

class QuantileCriticNet(nn.Module):
    """
    v1 QR-DQN critic: N_Q=64 FIXED quantile locations.
    More stable than IQN at 1000-episode budgets.
    """
    N_Q = 64
    def __init__(self,obs_d,hid=256):
        super().__init__()
        self.enc  = Encoder(obs_d,hid)
        self.head = _lin(hid,self.N_Q,gain=1.)
        self.register_buffer("taus",
            torch.tensor([(2*i-1)/(2*self.N_Q) for i in range(1,self.N_Q+1)]))
    def forward(self,x):      return self.head(self.enc(x))
    def mean(self,x):         return self.forward(x).mean(1)
    def cvar(self,x,alpha=0.25):
        q=self.forward(x); k=max(1,int(alpha*self.N_Q))
        return q.topk(k,dim=1,largest=False).values.mean(1)
    def qr_loss(self,pred,target):
        td  = target.unsqueeze(1)-pred
        hub = torch.where(td.abs()<=1,.5*td**2,td.abs()-.5)
        rho = (self.taus-(td.detach()<0).float()).abs()
        return (rho*hub).mean()


# =====================================================================
# 6.  RUNNING NORMALISER  (v1 unchanged)
# =====================================================================
class RMS:
    def __init__(self): self.m=0.; self.v=1.; self.n=1e-4
    def push(self,x):
        b,bv,bn=x.mean(),x.var(),len(x); tot=self.n+bn; d=b-self.m
        self.m+=d*bn/tot; self.v=(self.v*self.n+bv*bn+d**2*self.n*bn/tot)/tot; self.n=tot
    def norm(self,x): return (x-self.m)/(self.v**.5+1e-8)


# =====================================================================
# 7.  D-PPO AGENT  (v1 hyperparams -- proven)
# =====================================================================
class DPPO:
    """
    Distributional PPO -- v1 hyperparams restored:
      Actor:   PolicyNet (256-hid, 3 ResBlocks, 4-action head)
      Critic:  QuantileCriticNet (QR-DQN, 64 quantiles)
      lr_a     = 2e-4  (v1)
      lr_c     = 5e-4  (v1)
      eps_clip = 0.15  (v1, tighter = more conservative)
      ent_coef = 0.01  FIXED  (v1, stable exploration)
      cvar_alpha = 0.25 (v1)
      n_epochs = 8    (v1)
      batch    = 256  (v1, fits episode length ~80-200 steps)
      CosineAnnealingLR, no warm restart  (v1)
      CVaR ramp-up starts at 30% of training  (v1)
      Auxiliary urgency loss coef = 0.02  (v1)
    """
    def __init__(self, obs_d:int, act_d:int, T:int,
                 lr_a:float=2e-4, lr_c:float=5e-4,
                 gamma:float=0.995, lam:float=0.97,
                 eps_clip:float=0.15, ent_coef:float=0.01,
                 vf_coef:float=0.5, cvar_alpha:float=0.25,
                 n_epochs:int=8, batch:int=256):
        self.gamma=gamma; self.lam=lam; self.eps=eps_clip
        self.ent=ent_coef; self.vf=vf_coef; self.ca=cvar_alpha
        self.n_epochs=n_epochs; self.batch=batch
        self.step_count=0; self.T=T

        self.actor  = PolicyNet(obs_d, act_d).to(DEVICE)
        self.critic = QuantileCriticNet(obs_d).to(DEVICE)

        self.opt_a = optim.Adam(self.actor.parameters(),  lr=lr_a, eps=1e-5)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c, eps=1e-5)
        # v1: simple CosineAnnealingLR (no warm restart)
        self.sch_a = optim.lr_scheduler.CosineAnnealingLR(self.opt_a, T, 1e-5)
        self.sch_c = optim.lr_scheduler.CosineAnnealingLR(self.opt_c, T, 2e-5)
        self.rms   = RMS()

    @torch.no_grad()
    def act(self,obs:np.ndarray,mask:np.ndarray)->Tuple[int,float]:
        s=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
        lg,_=self.actor(s)
        lg=lg.masked_fill(~torch.tensor(mask).unsqueeze(0),-1e9)
        d=Categorical(logits=lg); a=d.sample()
        return int(a), float(d.log_prob(a))

    @torch.no_grad()
    def greedy(self,obs:np.ndarray,mask:np.ndarray)->int:
        s=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
        lg,_=self.actor(s)
        lg=lg.masked_fill(~torch.tensor(mask).unsqueeze(0),-1e9)
        return int(lg.argmax(1))

    def _gae(self,r,v,nv,d):
        adv=torch.zeros_like(r); g=0.
        for t in reversed(range(len(r))):
            delta=r[t]+self.gamma*nv[t]*(1-d[t])-v[t]
            g=delta+self.gamma*self.lam*(1-d[t])*g; adv[t]=g
        return adv, adv+v

    def update(self,buf:dict):
        self.step_count+=1
        R=torch.tensor(np.array(buf["r"]),dtype=torch.float32)
        self.rms.push(R.numpy())
        Rn=torch.tensor(self.rms.norm(R.numpy()),dtype=torch.float32)

        S =torch.tensor(np.array(buf["s"]),   dtype=torch.float32)
        NS=torch.tensor(np.array(buf["ns"]),  dtype=torch.float32)
        A =torch.tensor(np.array(buf["a"]),   dtype=torch.long)
        D =torch.tensor(np.array(buf["d"]),   dtype=torch.float32)
        LP=torch.tensor(np.array(buf["lp"]),  dtype=torch.float32)
        AT=torch.tensor(np.array(buf["at"]),  dtype=torch.float32)
        MK=torch.tensor(np.array(buf["mask"]),dtype=torch.bool)

        with torch.no_grad():
            v  = self.critic.mean(S)
            nv = self.critic.mean(NS)
            cv = self.critic.cvar(S,self.ca)

        adv,ret=self._gae(Rn,v,nv,D)
        # v1: CVaR ramp starts at 30% of training
        cw=0.10*min(1.,self.step_count/(self.T*0.3))
        adv=adv-cw*(v-cv).clamp(0).detach()
        adv=(adv-adv.mean())/(adv.std()+1e-8)

        N=len(S)
        for _ in range(self.n_epochs):
            idx=torch.randperm(N)
            for start in range(0,N,self.batch):
                mb=idx[start:start+self.batch]
                if len(mb)<4: continue
                s,a,rt,av,lp,at,mk=(S[mb],A[mb],ret[mb],adv[mb],LP[mb],AT[mb],MK[mb])

                lg,urg=self.actor(s)
                lg=lg.masked_fill(~mk,-1e9)
                dist=Categorical(logits=lg)
                nlp=dist.log_prob(a); ent=dist.entropy().mean()

                ratio=torch.exp(nlp-lp)
                obj=torch.min(ratio*av, ratio.clamp(1-self.eps,1+self.eps)*av)
                # v1: aux urgency loss coef=0.02
                a_loss=-obj.mean()-self.ent*ent+0.02*F.mse_loss(urg.squeeze(1),at)

                self.opt_a.zero_grad(); a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
                self.opt_a.step()

                c_loss=self.critic.qr_loss(self.critic(s),rt)
                self.opt_c.zero_grad(); (self.vf*c_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),0.5)
                self.opt_c.step()

        self.sch_a.step(); self.sch_c.step()

print("Agent ready.\n")


# =====================================================================
# 8.  TRAINING LOOP
#     v1 curriculum (80+ep//5, max 200) + v4 comp_hist diagnostics
# =====================================================================
def train(n_episodes:int=1000):
    print(f"Training D-PPO v5  |  episodes={n_episodes}  obs_dim={Env.OBS_DIM}\n")

    env   = Env(eval_mode=False)
    agent = DPPO(obs_d=env.observation_space.shape[0],
                 act_d=env.action_space.n,
                 T=n_episodes)

    rng_train = np.random.default_rng(SEED)
    ep_means  = []; ep_sums = []
    comp_hist = {"finish":[],"energy":[],"sla":[]}

    for ep in range(n_episodes):
        # v1 curriculum: slow ramp, max 200 tasks
        n_tasks = min(80 + ep//5, 200)
        tasks   = DS.window(n_tasks, rng_train)

        obs,_=env.reset(); env.load(tasks)
        buf={k:[] for k in ["s","a","r","ns","d","lp","at","mask"]}
        done=False; ep_r=0.; steps=0
        ep_comps={k:0. for k in comp_hist}

        while not done:
            mk    = env.mask()
            a, lp = agent.act(obs,mk)
            ns,r,done,_,_=env.step(a)

            buf["s"].append(obs);         buf["a"].append(a)
            buf["r"].append(r);           buf["ns"].append(ns)
            buf["d"].append(float(done)); buf["lp"].append(lp)
            buf["at"].append(1.-obs[3]);  buf["mask"].append(mk)

            for k,v in env._r_components.items():
                if k in ep_comps: ep_comps[k]+=v

            obs=ns; ep_r+=r; steps+=1

        ep_means.append(ep_r/max(steps,1))
        ep_sums.append(ep_r)
        for k in comp_hist: comp_hist[k].append(ep_comps[k]/max(steps,1))
        agent.update(buf)

        if (ep+1)%100==0:
            ec={k:np.mean(comp_hist[k][-100:]) for k in comp_hist}
            print(f"  Ep {ep+1:>5}/{n_episodes}  "
                  f"r/step {np.mean(ep_means[-100:]):+.4f}  "
                  f"[F:{ec['finish']:+.3f} E:{ec['energy']:+.3f} "
                  f"SLA:{ec['sla']:+.4f}]")

    print("\nTraining complete.\n")
    return agent, ep_means, comp_hist

agent, ep_rewards, comp_hist = train(n_episodes=1000)


# =====================================================================
# 9.  EVALUATION
# =====================================================================
def _eval_sched(sched, tasks):
    env=Env(eval_mode=True); env.reset()
    env.vm_fin[:]=0; env.vm_erg[:]=0; env.sla=0
    sched.run(env, copy.deepcopy(tasks))
    return env.makespan(), env.energy(), env.sla_viol()

def _eval_dppo(agent, tasks):
    env=Env(eval_mode=True); obs,_=env.reset()
    env.load(copy.deepcopy(tasks)); done=False
    while not done:
        a=agent.greedy(obs,env.mask())
        obs,_,done,_,_=env.step(a)
    return env.makespan(), env.energy(), env.sla_viol()

def compare(agent, n_windows=30, window_size=200):
    print("="*70)
    print(f"  COMPARISON  |  {n_windows} windows x {window_size} tasks")
    print(f"  Data: {DS.source}")
    print("="*70)

    rng=np.random.default_rng(SEED+1)
    scheds={"FCFS":FCFS(),"SJF":SJF(),"Round Robin":RoundRobin(),
            "EDF":EDF(),"MinMin":MinMin(),"MaxMin":MaxMin()}
    acc={nm:{"ms":[],"en":[],"sla":[]} for nm in list(scheds)+["D-PPO"]}

    for _ in range(n_windows):
        tasks=DS.window(window_size,rng)
        for nm,sc in scheds.items():
            ms,en,sla=_eval_sched(sc,tasks)
            acc[nm]["ms"].append(ms); acc[nm]["en"].append(en); acc[nm]["sla"].append(sla)
        ms,en,sla=_eval_dppo(agent,tasks)
        acc["D-PPO"]["ms"].append(ms); acc["D-PPO"]["en"].append(en); acc["D-PPO"]["sla"].append(sla)

    rows={}
    for nm,v in acc.items():
        rows[nm]={"Makespan":np.mean(v["ms"]),"Energy":np.mean(v["en"]),"SLA":np.mean(v["sla"]),
                  "_ms_all":v["ms"],"_en_all":v["en"],"_sla_all":v["sla"]}
    df_full=rows
    df=pd.DataFrame({k:{"Makespan":v["Makespan"],"Energy":v["Energy"],"SLA":v["SLA"]}
                      for k,v in rows.items()}).T

    print(f"\n{df.to_string(float_format='%.2f')}\n")
    dp=df.loc["D-PPO"]
    print("D-PPO vs best competing baseline:")
    print(f"  {'Metric':12} {'D-PPO':>9} {'Best':>9} {'Delta':>9} {'%':>7}  (makespan within 0.5% = tie)")
    print("  "+"-"*62)
    for m in ["Makespan","Energy","SLA"]:
        best=df.drop("D-PPO")[m].min(); d=dp[m]-best; p=d/(best+1e-9)*100
        mark="✓" if (m=="Makespan" and p<=0.5) or (m!="Makespan" and d<=0) else "~"
        print(f"  {mark} {m:12} {dp[m]:9.2f} {best:9.2f} {d:+9.2f} {p:+6.1f}%")
    return df, df_full

results, results_full = compare(agent)


# =====================================================================
# 10. VISUALISATIONS  (saved as PNGs to ./outputs/)
# =====================================================================
ALGOS    = list(results.index)
METRICS  = ["Makespan","Energy","SLA"]
PAL      = sns.color_palette("tab10", len(ALGOS))
ALGO_CLR = {a:(RED if a=="D-PPO" else PAL[i]) for i,a in enumerate(ALGOS)}

def _ax_style(ax,title,xlabel="",ylabel=""):
    ax.set_facecolor(PAN)
    ax.set_title(title,color="white",fontsize=12,fontweight="bold",pad=10)
    ax.set_xlabel(xlabel,color="#AAAAAA",fontsize=9)
    ax.set_ylabel(ylabel,color="#AAAAAA",fontsize=9)
    ax.tick_params(colors="#AAAAAA",labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#2A2A4A")
    ax.grid(True,alpha=0.2,color="#2A2A4A",linestyle="--")

def show(fig, name):
    """Save figure to outputs/ and print the path. No window is opened."""
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {name}.png  →  {os.path.abspath(path)}")


# Plot 1: Training curves + component breakdown
print("\n-- Plot 1: Training Curves")
fig,axes=plt.subplots(2,1,figsize=(14,9),facecolor=BG)
fig.subplots_adjust(hspace=0.4)
ax=axes[0]
raw=pd.Series(ep_rewards)
ax.fill_between(range(len(raw)),raw.values,alpha=0.08,color="#3498DB")
ax.plot(raw.rolling(50,min_periods=1).mean(),color="#3498DB",lw=1.8,label="50-ep avg")
ax.plot(raw.rolling(200,min_periods=1).mean(),color="#F39C12",lw=2.5,ls="--",label="200-ep avg")
_ax_style(ax,"D-PPO v5 Training Reward (per step)","Episode","Reward/step")
ax.legend(fontsize=9)
ax2=axes[1]
comp_cols={"finish":"#3498DB","energy":"#E74C3C","sla":"#E67E22"}
for k,c in comp_cols.items():
    s=pd.Series(comp_hist[k])
    ax2.plot(s.rolling(50,min_periods=1).mean(),color=c,lw=1.8,label=k.capitalize())
_ax_style(ax2,"Per-Component Reward (50-ep rolling mean)","Episode","Reward/step")
ax2.legend(fontsize=9)
fig.suptitle("D-PPO v5  |  v1 Core + v4 Shell",
             color="white",fontsize=13,fontweight="bold",y=1.01)
fig.tight_layout(); show(fig,"1_training_curves")


# Plot 2: Metric bars
print("-- Plot 2: Metric Bars")
fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor=BG)
fig.subplots_adjust(wspace=0.38)
for ax,m in zip(axes,METRICS):
    vals=results[m].values.astype(float); colors=[ALGO_CLR[a] for a in ALGOS]
    bars=ax.bar(ALGOS,vals,color=colors,edgecolor="#1A1A30",lw=0.8,alpha=0.9,zorder=3)
    ax.bar_label(bars,fmt="%.1f",padding=3,color="white",fontsize=7.5,fontweight="bold")
    _ax_style(ax,m,"",m)
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels(ALGOS,rotation=35,ha="right",fontsize=8.5,color="#CCCCCC")
    for bar,algo in zip(bars,ALGOS):
        if algo=="D-PPO": bar.set_edgecolor("white"); bar.set_linewidth(2.)
fig.suptitle(f"Scheduler Comparison -- {30} Evaluation Windows  |  {DS.source}",
             color="white",fontsize=13,fontweight="bold",y=1.02)
fig.tight_layout(); show(fig,"2_metric_bars")


# Plot 3: Box plots
print("-- Plot 3: Box Plots")
fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor=BG)
fig.subplots_adjust(wspace=0.38)
_key={"Makespan":"_ms_all","Energy":"_en_all","SLA":"_sla_all"}
for ax,m in zip(axes,METRICS):
    ld=[{"Algorithm":a,m:v} for a in ALGOS for v in results_full[a][_key[m]]]
    ldf=pd.DataFrame(ld)
    sns.boxplot(data=ldf,x="Algorithm",y=m,palette=ALGO_CLR,
                width=0.55,linewidth=0.9,fliersize=2.5,ax=ax,order=ALGOS)
    _ax_style(ax,f"{m} Distribution","Algorithm",m)
    ax.set_xticklabels(ALGOS,rotation=35,ha="right",fontsize=8.5,color="#CCCCCC")
fig.suptitle("Per-Window Distribution",color="white",fontsize=13,fontweight="bold",y=1.02)
fig.tight_layout(); show(fig,"3_boxplots")


# Plot 4: Improvement heatmap
print("-- Plot 4: Improvement Heatmap")
bases=[a for a in ALGOS if a!="D-PPO"]
drow=results.loc["D-PPO",METRICS].astype(float)
heat=np.array([[100*(float(results.loc[b,m])-float(drow[m]))/max(abs(float(results.loc[b,m])),1e-9)
                for m in METRICS] for b in bases])
heat_df=pd.DataFrame(heat,index=bases,columns=METRICS)
fig,ax=plt.subplots(figsize=(9,5),facecolor=BG)
ax.set_facecolor(PAN)
sns.heatmap(heat_df,annot=True,fmt="+.1f",cmap="RdYlGn",center=0,vmin=-30,vmax=80,
            linewidths=0.5,linecolor="#1A1A30",
            annot_kws={"size":11,"weight":"bold","color":"white"},
            ax=ax,cbar_kws={"shrink":0.8})
ax.set_title("D-PPO Improvement % vs Each Baseline",
             color="white",fontsize=12,fontweight="bold",pad=12)
ax.tick_params(colors="#CCCCCC",labelsize=10)
for _,sp in ax.spines.items(): sp.set_edgecolor("#2A2A4A")
ax.collections[0].colorbar.ax.tick_params(colors="#CCCCCC")
fig.tight_layout(); show(fig,"4_heatmap")


# Plot 5: Radar
print("-- Plot 5: Radar")
ndf=results[METRICS].astype(float).copy()
for m in METRICS:
    mx=ndf[m].max(); ndf[m]=ndf[m]/mx if mx>0 else ndf[m]
K=len(METRICS); angles=[k/K*2*np.pi for k in range(K)]+[0]
fig=plt.figure(figsize=(8,8),facecolor=BG)
ax=fig.add_subplot(111,polar=True); ax.set_facecolor(PAN)
ax.set_title("Normalised Radar (lower=better)",color="white",fontsize=12,
             fontweight="bold",y=1.08)
for i,algo in enumerate(ALGOS):
    row=list(ndf.loc[algo].values)+[ndf.loc[algo].values[0]]
    c=ALGO_CLR[algo]; lw=3. if algo=="D-PPO" else 1.2
    ax.plot(angles,row,color=c,lw=lw,label=algo)
    ax.fill(angles,row,color=c,alpha=0.22 if algo=="D-PPO" else 0.04)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(METRICS,color="white",fontsize=10)
ax.tick_params(colors="#666688",labelsize=8); ax.spines["polar"].set_edgecolor("#2A2A4A")
ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.2),
          facecolor=PAN,labelcolor="white",fontsize=9,edgecolor="#2A2A4A")
fig.tight_layout(); show(fig,"5_radar")


# Plot 6: Violin + swarm
print("-- Plot 6: Violin + Swarm")
fig,axes=plt.subplots(1,2,figsize=(16,6),facecolor=BG)
fig.subplots_adjust(wspace=0.4)
for ax,m,key in zip(axes,["Energy","SLA"],["_en_all","_sla_all"]):
    ld=[{"Algorithm":a,m:v} for a in ALGOS for v in results_full[a][key]]
    ldf=pd.DataFrame(ld)
    sns.violinplot(data=ldf,x="Algorithm",y=m,palette=ALGO_CLR,
                   inner="quartile",linewidth=0.9,ax=ax,order=ALGOS,
                   cut=0,density_norm="width")
    sns.stripplot(data=ldf,x="Algorithm",y=m,color="white",
                  alpha=0.35,size=2.5,ax=ax,jitter=True,order=ALGOS)
    _ax_style(ax,f"{m} -- Violin + Swarm","Algorithm",m)
    ax.set_xticklabels(ALGOS,rotation=35,ha="right",fontsize=8.5,color="#CCCCCC")
fig.suptitle("Energy & SLA Distribution",color="white",fontsize=13,fontweight="bold",y=1.02)
fig.tight_layout(); show(fig,"6_violin_swarm")


results.to_csv(os.path.join(OUT_DIR,"results.csv"))
print(f"\nResults CSV -> {OUT_DIR}/results.csv")
print("All done!")
