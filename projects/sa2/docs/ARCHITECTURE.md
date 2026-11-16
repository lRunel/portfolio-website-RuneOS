# Architecture: DRL-Driven Hybrid Sync-Async Distributed Deep Learning

> **Reading order:** This document is written **bottom-to-top** — starting from the
> lowest-level primitives and building up to the full system behaviour.  
> Read each section before the next so every concept is grounded before it is used.

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)  
2. [Layer 0 — Data Primitives (`staleness_filter.py :: GradientPacket`)](#2-layer-0--data-primitives)  
3. [Layer 1 — Staleness Filter (`staleness_filter.py :: StalenessFilter`)](#3-layer-1--staleness-filter)  
4. [Layer 2 — DRL Orchestrators (`drl_orchestrator.py` / `sac_orchestrator.py`)](#4-layer-2--drl-orchestrators)  
5. [Layer 3 — Heuristic Pre-Training (`heuristic_pretrain.py`)](#5-layer-3--heuristic-pre-training)  
6. [Layer 4 — Async Buffer & Background Listener](#6-layer-4--async-buffer--background-listener)  
7. [Layer 5 — Hybrid Controller (`hybrid_controller.py`)](#7-layer-5--hybrid-controller)  
8. [Layer 6 — HTTP API (`controller.py`)](#8-layer-6--http-api)  
9. [End-to-End Round Walk-Through](#9-end-to-end-round-walk-through)  
10. [Reward Function Deep Dive](#10-reward-function-deep-dive)  
11. [Communication Optimisations](#11-communication-optimisations)  
12. [Configuration & Tuning Guide](#12-configuration--tuning-guide)  
13. [File Map](#13-file-map)  

---

## 1. Motivation & Problem Statement

### The Challenge: Heterogeneous Devices

A traditional distributed training setup assumes workers of roughly equal speed.
In an **opportunistic** setting — phones, laptops, desktops on a shared network —
you face a fundamental tension:

| Approach | Problem |
|---|---|
| **Pure Synchronous** (wait for all workers) | Fastest worker idles waiting for the slowest (the "straggler problem"). Throughput collapses. |
| **Pure Asynchronous** (apply every gradient immediately) | Old ("stale") gradients from slow workers corrupt the model. Divergence is common. |

### The Solution: Time-Based Hybrid

The framework combines the best of both worlds:

```
Open collection window for T_optimal seconds  (ASYNC phase)
 Workers submit at their own pace
 Every submission is buffered in a queue

When T_optimal expires - SYNC cutoff
 Drain the queue instantly
 Filter stale gradients (dynamic S_max threshold)
 Apply Top-K sparsification (DRL-controlled S_ratio)
 Aggregate and update the model atomically
```

**T_optimal, S_ratio, and S_max are not fixed.** A Deep Reinforcement Learning (DRL) agent
learns to set all three dynamically based on worker speeds, network conditions, and model state.
This 3D action space is the key innovation.

---

## 2. Layer 0 — Data Primitives

**File:** `controller/staleness_filter.py`  
**Class:** `GradientPacket`

Every gradient submission is wrapped in a `GradientPacket` before it touches
any other system. This is a pure data container with no logic.

```python
@dataclass
class GradientPacket:
    worker_id         : str                  # "phone-a3f2" etc.
    gradients         : list[torch.Tensor]   # one tensor per model parameter
    worker_version    : int                  # global version the worker trained on
    samples_processed : int                  # data samples used - for weighted avg
    received_at       : float                # unix timestamp (auto-set)
```

**Why does `worker_version` matter?**  
The controller increments `global_version` every round. If a worker downloaded
the model at version 10 and the controller is now at version 15, this gradient is
5 rounds stale. The filter uses this number to decide how much to trust it.

**Why `samples_processed`?**  
A worker that processed 1 000 samples carries more information than one that
processed 100. During aggregation, the contribution of each gradient is weighted
by both its freshness *and* how many samples it represents.

---

## 3. Layer 1 — Staleness Filter

**File:** `controller/staleness_filter.py`  
**Class:** `StalenessFilter`

### What is Staleness?

```
staleness = global_version_now - packet.worker_version
```

A staleness of 0 means the gradient was computed on the *current* model — perfect.
A staleness of 3 means the model moved on by 3 rounds since this gradient was computed.

### Soft-Weight Scheme

Instead of a binary accept/reject, the filter assigns a **continuous weight**:

- `weight = 1.0 - (staleness / S_max)` for 0 <= staleness <= S_max (floored at 0.01)
- `weight = 0.0` for staleness > S_max (hard dropped)

**Dynamic S_max:** The staleness threshold is no longer fixed — it is controlled
per round by the DRL agent via the `dynamic_max_staleness` parameter. The agent
learns to set S_max loose in early training (every gradient helps) and tight
during fine-tuning (only fresh gradients improve accuracy).

### Weighted Average (Aggregation Helper)

After filtering, `StalenessFilter.weighted_average(valid_results)` computes:

```
agg_grad_i = Sum_j  (weight_j x samples_j / Sum_k weight_k*samples_k)  x  grad_i_j
```

This is an **importance-weighted** federated average. Fresh gradients from
large-batch workers dominate; stale small-batch workers contribute proportionally less.

---

## 4. Layer 2 — DRL Orchestrators

The framework supports two DRL algorithms via the `DRL_ALGORITHM` environment variable:
**PPO** (default, on-policy) and **SAC** (off-policy, more sample-efficient).

Both share the same public interface: `build_state()`, `act()`, `store_reward()`, `update()`.

### 4.1 Markov Decision Process (MDP) Formulation

| MDP Component | Value |
|---|---|
| **State S** | 5-dim float vector |
| **Action A** | 3-dim continuous: (T_optimal, S_ratio, S_max) |
| **Reward R** | α·ΔAcc − β·T − γ·n_dropped + δ·(1−S_ratio) − ε·(S_max/SMAX_MAX) |
| **Policy π** | Gaussian (PPO) or Squashed Gaussian (SAC) |

### 4.2 State Vector

```
index  feature               normalisation           meaning
  0    avg_compute           / 5000.0                mean samples/sec across workers
  1    avg_bandwidth         / 50_000_000.0           mean bytes/sec across workers
  2    n_workers             / 10.0                   normalised worker count
  3    current_accuracy      / 100.0                  last measured test accuracy [0,1]
  4    staleness_ratio       [0, 1]                   fraction of last round's grads dropped
```

### 4.3 Action Space (ACTION_DIM = 3)

| Dim | Action | Range | Purpose |
|-----|--------|-------|---------|
| 0 | T_optimal | [0.2, 20.0] seconds | Collection window duration |
| 1 | S_ratio | [0.30, 1.0] | Gradient sparsification ratio (fraction kept) |
| 2 | S_max | [1.0, 10.0] | Dynamic staleness threshold |

### 4.4 PPO Orchestrator (`drl_orchestrator.py`)

**Actor:** `_MLP(5,64,Tanh) → mean_head(sigmoid-scale) + log_std_head(clamp[-4,1])`  
**Critic:** `_MLP(5,64,Tanh) → value_head → V(s)`

The sigmoid-scale trick ensures the *mean* always falls in valid ranges while keeping
the raw network output unconstrained (avoids exploding gradients at boundaries).

PPO update runs 4 epochs per collected sample with clipped surrogate objective +
entropy bonus for exploration.

### 4.5 SAC Orchestrator (`sac_orchestrator.py`)

**Actor:** `ReLU backbone(128) → SquashedGaussianActor → tanh(N(μ,σ)) → rescale`  
**Twin Critic:** Two independent `Q(s,a)` networks (prevents overestimation bias)  
**Target Networks:** Slowly-updated critic copies via Polyak averaging (τ = 0.005)  
**Replay Buffer:** Circular buffer of 25,000 (s,a,r,s',done) tuples  
**Temperature α:** Learned automatically — no manual entropy tuning

SAC advantages over PPO for this system:
- **Off-policy:** reuses past experience via replay buffer → more sample efficient
- **Automatic exploration:** learned temperature α replaces manual ENTROPY_C tuning
- **Tanh squashing:** native bounded continuous actions with Jacobian-corrected log-probs
- **Pre-fillable replay buffer:** can be warm-started with 20,000 heuristic tuples

---

## 5. Layer 3 — Heuristic Pre-Training

**File:** `controller/heuristic_pretrain.py`

### The Heuristic Oracle

Before DRL, the framework used hand-crafted scheduling rules. These are codified
in `HeuristicOracle`, which computes optimal actions from state:

- **T_optimal:** Based on expected task time + bandwidth pressure + worker count
- **S_ratio:** Driven by communication pressure (transmit/compute ratio) and training phase
- **S_max:** Accuracy-phase-aware (loose early, tight late) + staleness feedback

### Behavioral Cloning

`pretrain_actor()` trains any Actor network (PPO or SAC) to imitate the oracle
via supervised MSE loss on 2,000–20,000 synthetic states. This provides a warm
start so the DRL agent doesn't explore catastrophically in early rounds.

### Replay Buffer Warm-Start (SAC only)

`generate_replay_data()` creates synthetic (s, a, r, s', done) tuples using the
oracle, then fills the SAC replay buffer. This eliminates the cold-start problem
— SAC can begin learning from round 1 with 20,000 samples already in its buffer.

---

## 6. Layer 4 — Async Buffer & Background Listener

**Location:** module-level in `controller/controller.py`

```python
gradient_queue: asyncio.Queue = asyncio.Queue()
```

This single line is the heart of the hybrid architecture:

```
Worker POST /send_gradients
        |
        v
  await gradient_queue.put(packet)    <- non-blocking, O(1)
        |
        v
  return {"status": "queued"}         <- response to worker in ~1ms

  Meanwhile, HybridController is sleeping T_optimal seconds
  The queue grows as workers submit

  When sleep expires -> _drain_queue()
  All packets collected in one atomic sweep
```

`asyncio.Queue` is used (not `threading.Queue`) because both the HTTP endpoints
and the training loop run on the *same* asyncio event loop — coroutine-safe with
zero threading overhead.

---

## 7. Layer 5 — Hybrid Controller

**File:** `controller/hybrid_controller.py`  
**Class:** `HybridController`

### 7.1 Constructor

```python
HybridController(
    model, optimizer, scheduler, evaluate_fn, gradient_queue,
    max_staleness, device, target_samples, dataset_size,
    dashboard,           # live terminal UI
    drl_algorithm="ppo", # "ppo", "sac", or "heuristic"
    skip_pretrain=False,
)
```

At construction, the controller selects the DRL backend and runs behavioral
cloning pre-training (unless `skip_pretrain=True` or a SAC checkpoint is loaded).

### 7.2 Round Life-Cycle

```
ROUND N

  1. OBSERVE    build_state(scheduler, accuracy, staleness_ratio)
                -> 5-dim state tensor

  2. ACT        drl.act(state)
                -> (T_optimal, S_ratio, S_max) — 3D action tuple

  3. WAIT       await asyncio.sleep(T_optimal)
                During sleep, HTTP endpoint keeps enqueuing grads

  4. DRAIN      _drain_queue()   [non-blocking get_nowait() loop]
                -> list of GradientPackets

  5. FILTER     filter.filter_batch(packets, global_version,
                                     dynamic_max_staleness=S_max)
                -> (valid_results, dropped_results)

  6. AGGREGATE  StalenessFilter.weighted_average(valid_results)
                _apply_gradients(agg_grads)  [under _grad_lock]
                Store weight deltas in history_buffer for delta sync

  7. VERSION    global_version += 1  (before evaluation)

  8. EVALUATE   await evaluate_fn()  -> new_accuracy

  9. REWARD     R = α·ΔAcc − β·T − γ·n_dropped + δ·(1−S_ratio) − ε·(S_max/SMAX_MAX)
                drl.store_reward(R)
                drl.update()  -> PPO or SAC step
```

### 7.3 FedAvg-Style Pseudo-Gradient Application

The optimizer is `SGD(lr=1.0)` — not Adam. Workers compute pseudo-gradients
(accumulated gradients normalised by step count), and the controller applies
them directly with `optimizer.step()`. This is equivalent to Federated Averaging:
workers essentially perform local SGD steps, and the controller averages the
resulting weight changes.

---

## 8. Layer 6 — HTTP API

**File:** `controller/controller.py`

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/register` | Register worker with hardware profile |
| `GET` | `/get_task` | Pull latest model + data batch (supports delta sync) |
| `POST` | `/send_gradients` | Enqueue a sparsified gradient submission |
| `GET` | `/status` | Live controller snapshot |

### Delta Sync in `/get_task`

The endpoint supports three model update types:
- **`full`:** Complete state dict (first connection or large version gap)
- **`delta`:** Sum of weight deltas from worker's version to current (bandwidth-efficient)
- **`none`:** Worker is already up-to-date (zero model data sent)

### `/send_gradients` — Gradient Sparsification

Workers apply Top-K sparsification before sending, controlled by `S_ratio` from
the DRL agent. The controller deserialises sparse gradients and reconstructs
dense tensors before queuing.

---

## 9. End-to-End Round Walk-Through

```
Global version = 10, DRL algorithm = SAC

1. SAC observes: avg_compute=200 s/s, 3 workers, accuracy=71.2%, staleness_ratio=0.05
   Actor samples: T_optimal=5.8s, S_ratio=0.65, S_max=5

2. asyncio.sleep(5.8) begins

   t=0.5s  Desktop  submits sparsified gradients (v=10, staleness=0)  -> enqueued
   t=1.9s  Laptop   submits sparsified gradients (v=9,  staleness=1)  -> enqueued
   t=6.2s  Phone    submits gradients (v=8, staleness=2) -> MISSED WINDOW

   t=5.8s  asyncio.sleep returns

3. _drain_queue() -> [DesktopPacket(v=10), LaptopPacket(v=9)]

4. StalenessFilter (dynamic S_max=5):
   Desktop: staleness=0  -> weight=1.00   valid
   Laptop:  staleness=1  -> weight=0.80   valid   (1 - 1/5 = 0.80)
   dropped=0

5. weighted_average() + optimizer.step()
   Store delta in history_buffer[10] for future delta sync
   global_version = 11

6. evaluate_model() -> accuracy = 72.5%
   acc_improvement = 72.5 - 71.2 = 1.3

7. R = 1.0×1.3 − 0.05×5.8 − 0.2×0 + 0.5×(1−0.65) − 0.05×(5/10)
     = 1.3 − 0.29 − 0 + 0.175 − 0.025 = 1.16
   SAC stores in replay buffer, samples mini-batch, updates critics+actor+α

8. Phone's pending gradient (v=8) will have staleness=11-8=3 next round
```

---

## 10. Reward Function Deep Dive

```
R = α · ΔAccuracy  −  β · T_optimal  −  γ · n_dropped  +  δ · (1 − S_ratio)  −  ε · (S_max / SMAX_MAX)
```

### Coefficients

| Symbol | Default | Controls |
|---|---|---|
| `α` (ALPHA) | `1.0` | How much the agent values model improvement |
| `β` (BETA) | `0.05` | Penalty per second of T_optimal — pushes toward shorter windows |
| `γ` (GAMMA_PENALTY) | `0.2` | Penalty per dropped gradient — forces inclusion of slower workers |
| `δ` (SPARSITY_BONUS) | `0.5` | Bonus for gradient compression — rewards bandwidth efficiency |
| `ε` (SMAX_PENALTY) | `0.05` | Mild penalty for loose staleness — prevents always maxing S_max |

### Term Interactions

The sparsity bonus is gated: `δ·(1−S_ratio)` is only awarded if accuracy didn't
regress this round (acc_improvement ≥ −0.1). This prevents the agent from
compressing too aggressively at the cost of model quality.

The `ε` penalty prevents a degenerate equilibrium where the agent always sets
S_max to the maximum to avoid the `γ` penalty — there's a cost to being too
tolerant of stale gradients.

---

## 11. Communication Optimisations

### Delta Sync (Incremental Model Updates)

The controller maintains a `history_buffer` of weight deltas (last 50 versions).
When a worker requests `/get_task` with a recent `model_version`, the controller
sums the intermediate deltas and sends them instead of the full state dict.
This reduces payload from ~1.6 MB to ~10–100 KB for small updates.

### Top-K Gradient Sparsification

Workers apply `sparsify_gradients()` before sending: only the top `S_ratio`
fraction of gradient values (by magnitude) are transmitted. The rest are zeroed.
At S_ratio=0.30, this reduces gradient payload by 70%.

The controller reconstructs dense tensors via `desparsify_gradients()`.

---

## 12. Configuration & Tuning Guide

All training parameters live in the YAML config files under `configs/`.

```yaml
training:
  learning_rate: 0.001   # Adam lr for the global model
  batch_size:    64       # samples per gradient step
  target_epochs: 5        # total training budget
  max_staleness: 5        # S_max initial value (DRL overrides dynamically)
  steps_per_task: 15      # steps per worker task
```

### DRL Algorithm Selection

Set via environment variable `DRL_ALGORITHM`:

| Value | Description |
|---|---|
| `ppo` (default) | PPO Actor-Critic, on-policy, simple but sample-inefficient |
| `sac` | SAC with twin critics + replay buffer, off-policy, sample-efficient |
| `heuristic` | Pure rule-based scheduling, no learning |

### PPO Constants (`drl_orchestrator.py`)

| Constant | Default | Description |
|---|---|---|
| `T_MIN` / `T_MAX` | `0.2s` / `20.0s` | Collection window bounds |
| `S_MIN` / `S_MAX` | `0.30` / `1.0` | Sparsity ratio bounds |
| `SMAX_MIN` / `SMAX_MAX` | `1.0` / `10.0` | Dynamic staleness bounds |
| `PPO_EPOCHS` | `4` | PPO gradient steps per round |
| `CLIP_EPS` | `0.2` | PPO clip ratio |
| `ENTROPY_C` | `0.01` | Exploration bonus |

### SAC Constants (`sac_orchestrator.py`)

| Constant | Default | Description |
|---|---|---|
| `SAC_TAU` | `0.005` | Soft target update rate (Polyak averaging) |
| `SAC_BATCH_SIZE` | `64` | Mini-batch for replay sampling |
| `SAC_BUFFER_SIZE` | `25,000` | Replay buffer capacity |
| `SAC_TARGET_ENTROPY` | `-3.0` | Target entropy (−ACTION_DIM) |

---

## 13. File Map

```
distributed-dl/
├── controller/
│   ├── __init__.py               <- Package exports
│   ├── controller.py             <- FastAPI app, HTTP endpoints, lifespan hook
│   ├── hybrid_controller.py      <- Main async training loop (9-step round)
│   ├── drl_orchestrator.py       <- PPO Actor-Critic, 3D action policy
│   ├── sac_orchestrator.py       <- SAC: twin critics, replay buffer, auto-α
│   ├── heuristic_pretrain.py     <- Behavioral cloning, synthetic replay data
│   ├── staleness_filter.py       <- GradientPacket, StalenessFilter, weighted avg
│   ├── scheduler.py              <- Worker registry, hardware stats, auto-purge
│   └── dashboard.py              <- Live ANSI terminal dashboard
│
├── worker/
│   ├── worker.py                 <- WorkerNode: delta sync + sparsified gradients
│   └── trainer.py                <- Local training engine (gradient-only, no optimizer)
│
├── common/
│   ├── config.py                 <- YAML config loader, defaults
│   └── serialization.py          <- Base64 encode/decode + Top-K sparsification
│
├── models/                       <- CNN, MLP, RNN, Transformer, NeuralNet, CIFAR-CNN
├── datasets/                     <- MNIST, CIFAR-10 loaders
├── configs/                      <- Per-experiment YAML files
├── tests/                        <- Test suite (7 scripts + graph generation)
├── pretrain_sac.py               <- Offline SAC pre-training (20,000 examples)
├── docs/
│   ├── ARCHITECTURE.md           <- This document
│   ├── TEXTBOOK_PART1.md         <- Theoretical foundations
│   ├── TEXTBOOK_PART2.md         <- Annotated source walk-through
│   ├── DRL_COMPARISON_REPORT.md  <- PPO vs SAC analysis
│   ├── EVALUATION_REPORT.md      <- Benchmarking methodology & results
│   ├── STRENGTHS_AND_WEAKNESSES.md <- Critical framework analysis
│   └── TESTING_PROGRESS.md       <- Test execution status
└── normal_train.py               <- Baseline single-machine training
```

### Data Flow Summary

```
Worker --POST /send_gradients--> gradient_queue (asyncio.Queue)
           (sparsified Top-K)           |
                                        | (buffered during T_optimal)
                                        |
                                        v
                                 HybridController._drain_queue()
                                        |
                                        v
                                 desparsify_gradients() → dense tensors
                                        |
                                        v
                                 StalenessFilter.filter_batch(dynamic_max_staleness=S_max)
                                        |
                            +───────────┴──────────+
                        valid (weighted)        dropped (counted)
                            |                       |
                            v                       v
                    weighted_average()         -> DRL γ penalty
                            |
                            v
                     optimizer.step() (SGD lr=1.0, FedAvg-style)
                            |
                            v
                     store delta in history_buffer (for delta sync)
                            |
                            v
                     global_version += 1
                            |
                            v
                     evaluate_model()
                            |
                            v
                     R = α·ΔAcc − β·T − γ·drops + δ·(1−S_ratio) − ε·(S_max/MAX)
                            |
                            v
                     DRL.update()  (PPO or SAC step)
                            |
                            +--------------------> next round
```
