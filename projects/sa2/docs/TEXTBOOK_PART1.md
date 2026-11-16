# Opportunistic Distributed Deep Learning — Complete Technical Textbook

**Project:** `distributed-dl`
**Stack:** Python 3.10+, PyTorch, FastAPI, asyncio
**Purpose:** Train neural networks across heterogeneous personal devices (phones, laptops, desktops) using a DRL-driven hybrid sync-async parameter server.

---

## PART 1 — THEORETICAL FOUNDATIONS

---

### Chapter 1: Why Distributed Training Exists

#### 1.1 The Compute Problem

Deep learning models learn by repeatedly computing a *gradient* — the direction in which to adjust millions of numbers (parameters/weights) so that the model makes fewer mistakes. A single forward + backward pass over a mini-batch costs:

```
Time ∝ (batch_size × model_parameters × dataset_size) / hardware_flops
```

For large datasets and models, one machine training for days or weeks becomes impractical. The solution is to split the work across multiple machines in parallel.

#### 1.2 The Two Parallelism Strategies

**Data Parallelism** — Each machine gets the same model but a different chunk of data. All machines compute gradients independently, then those gradients are averaged and used to update a single shared model. This is what this framework does.

**Model Parallelism** — The model itself is split across machines. Different layers run on different devices. This project does NOT use model parallelism.

#### 1.3 The Parameter Server Pattern

This project uses the **Parameter Server (PS)** pattern:

```
Workers (many)          Controller (one = parameter server)
──────────────          ─────────────────────────────────
Fetch model     ◄──────  Broadcast current weights
Train locally
Compute grads   ──────►  Receive and aggregate gradients
                         Update global model
Repeat
```

The controller owns the canonical model. Workers are stateless — they receive a fresh model copy each task, compute gradients on local data, and send those gradients back. They do not keep any persistent model state between tasks.

#### 1.4 The Straggler Problem and Staleness

In a **synchronous** PS, the controller waits for ALL workers before updating. One slow device (a phone on WiFi) blocks every fast device (a desktop). Throughput collapses to the slowest participant.

In a **purely asynchronous** PS, gradients are applied the moment they arrive. A phone that finished computing on model version 5 while the server is now at version 20 submits a gradient that points in a direction computed from a 15-step-old model. This is **staleness**, and it causes:
- Gradient directions that contradict the current loss landscape
- Model divergence or oscillation
- Reduced final accuracy

**Staleness** is formally defined as:
```
staleness = global_version_at_receipt - worker_version_at_compute_time
```

#### 1.5 The Hybrid Solution

This project's answer: a **time-based hybrid**. For each training round:
1. Open a collection window of T seconds (async — workers submit freely)
2. When T expires, close the window and aggregate everything that arrived (sync barrier)
3. Discard or downweight submissions that are too stale

The key insight: **T is not fixed**. A Deep Reinforcement Learning agent learns the optimal T dynamically per round.

---

### Chapter 2: Gradient Descent and Federated Averaging

#### 2.1 Standard SGD

The fundamental update rule for a parameter θ with learning rate η:
```
θ ← θ - η · ∇_θ L(θ; x, y)
```
where L is the loss function (cross-entropy in this project), x is the input batch, y is the labels.

#### 2.2 Gradient Accumulation Across Workers

When K workers each compute gradients on different data:
```
g_final = (1/K) Σ_k  g_k
```

This is mathematically equivalent to computing the gradient over the combined dataset of all workers — the foundation of distributed training correctness.

#### 2.3 Importance-Weighted Averaging

This project does NOT use a simple uniform average. It uses importance weighting:
```
g_final = Σ_k  (w_k × n_k / Σ_j w_j × n_j) × g_k
```
where:
- `w_k` = staleness weight (0.01 to 1.0, based on gradient age)
- `n_k` = samples_processed by worker k

A gradient from a worker that processed 1000 samples on fresh data counts much more than one from a worker that processed 64 samples on 4-round-old data.

#### 2.4 The Staleness Weight Formula

```
w = 1.0 - (staleness / S_max)    if 0 ≤ staleness ≤ S_max
w = 0.0                          if staleness > S_max  (hard dropped)
```

This creates a linear decay: fresh gradients get full weight, barely-accepted gradients (staleness = S_max) get near-zero weight (~0.01 floor).

---

### Chapter 3: Deep Reinforcement Learning for Scheduling

#### 3.1 What is Reinforcement Learning?

An **agent** interacts with an **environment** by taking **actions** and receiving **rewards**. The goal is to learn a **policy** π(a|s) — a mapping from states to actions — that maximises the expected cumulative reward over time.

```
Agent --action(a)--> Environment --reward(r), next_state(s')--> Agent
```

In this project:
- **Agent** = DRLOrchestrator
- **Environment** = the distributed training system (workers, network, model)
- **Action** = T_optimal (how many seconds to wait)
- **State** = snapshot of worker stats + model accuracy + staleness history
- **Reward** = a scalar combining accuracy gain, latency cost, and fairness penalty

#### 3.2 Markov Decision Process (MDP)

A formal MDP is defined by the tuple (S, A, R, P, γ):
- **S** — State space (5-dim continuous vector)
- **A** — Action space (1-dim continuous: T_optimal ∈ [0.5, 30.0])
- **R** — Reward function R(s, a, s')
- **P** — Transition probability P(s'|s,a) — implicit (the training system)
- **γ** — Discount factor (0.99) — how much future rewards matter

The MDP is **Markovian**: the current state contains all information needed to choose the next action. Past history is irrelevant given the current state.

#### 3.3 Policy Gradient Methods

Instead of learning Q-values (like DQN), policy gradient methods directly parameterise and optimise the policy π_θ(a|s). The REINFORCE objective is:

```
J(θ) = E_τ~π_θ [ Σ_t r_t ]
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

where G_t is the discounted return from time t:
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

The intuition: if an action led to high return (G_t is large), increase the log-probability of taking that action. If it led to low return, decrease it.

#### 3.4 The Variance Problem and Actor-Critic

Raw REINFORCE has high variance — return estimates fluctuate wildly, making learning unstable. The fix: subtract a **baseline** V(s) (the expected value of the state) from the return to get the **advantage**:

```
A_t = G_t - V(s_t)
```

A_t > 0 means "this action did better than expected" → increase its probability.
A_t < 0 means "this action did worse than expected" → decrease its probability.

The **Actor-Critic** architecture uses two networks:
- **Actor**: learns π_θ(a|s) — the policy
- **Critic**: learns V_φ(s) — the value function baseline

#### 3.5 SAC: Soft Actor-Critic

SAC (Schulman et al., 2017) adds a safety constraint: the new policy cannot be too far from the old policy. This prevents catastrophic updates that destroy previously learned behaviour.

The clipped surrogate objective:
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   ← probability ratio

L_CLIP(θ) = E_t [ min(r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```

If r_t > 1+ε, the action is being made too much more likely → clip.
If r_t < 1-ε, the action is being made too much less likely → clip.

The overall SAC loss with entropy bonus H:
```
L(θ) = -L_CLIP(θ) - c_entropy · H(π_θ)
```

The entropy bonus H encourages exploration — it penalises the policy for becoming too deterministic too early.

#### 3.6 Continuous Action Spaces and the Gaussian Policy

This project's action (T_optimal) is a real number, not a discrete choice. For continuous actions, the policy is a **Gaussian distribution**:

```
π_θ(a|s) = N(μ_θ(s), σ_θ(s))
```

The actor network outputs μ (mean) and log σ (log standard deviation). To sample an action:
1. `a ~ N(μ, σ)` via reparameterisation trick: `a = μ + σ·ε` where `ε ~ N(0,1)`
2. Clip to valid range [T_MIN, T_MAX]

The **sigmoid-scale trick** in the Actor network:
```python
mean = torch.sigmoid(raw_output) * (T_MAX - T_MIN) + T_MIN
```
This squashes the mean into [T_MIN, T_MAX] without clipping the gradient — sigmoid is differentiable everywhere, unlike hard clipping.

---

### Chapter 4: Asyncio and Concurrent Programming

#### 4.1 The GIL and Why Threads Don't Help for CPU Work

Python's Global Interpreter Lock (GIL) means only one thread executes Python bytecode at a time. For CPU-bound work (gradient computation), multiple threads don't give true parallelism. However, for **I/O-bound** work (waiting for HTTP responses, sleeping), threads release the GIL and do run concurrently.

#### 4.2 asyncio: Cooperative Concurrency

`asyncio` is Python's event loop framework. It uses **coroutines** — functions that can voluntarily pause their execution at `await` points, letting other coroutines run:

```python
async def my_coroutine():
    result = await some_io_operation()  # pause here, let others run
    return result
```

This is **cooperative** (not preemptive) — a coroutine runs until it explicitly yields at an `await`. There is no forced context switching. One thread handles many concurrent tasks efficiently.

#### 4.3 asyncio.Queue as a Thread-Safe Channel

`asyncio.Queue` is a FIFO queue safe to use between coroutines on the same event loop:

```python
q = asyncio.Queue()

# Producer (HTTP endpoint coroutine)
await q.put(item)          # non-blocking, always returns immediately

# Consumer (training loop coroutine)
item = await q.get()       # blocks until item available
item = q.get_nowait()      # raises QueueEmpty immediately if empty
```

`get_nowait()` is the "drain" pattern — it empties the queue without waiting for new items.

#### 4.4 asyncio.create_task() — Fire and Forget

```python
task = asyncio.create_task(my_coroutine())
```

Schedules `my_coroutine()` to run concurrently on the current event loop. The calling coroutine does NOT wait for it — it continues immediately. In this project, the training loop is started this way at server startup.

#### 4.5 The FastAPI + asyncio Integration

FastAPI is built on Starlette/asyncio. All `async def` endpoints run as coroutines on the same event loop. This means:
- The HTTP server and the training loop share one event loop thread
- No threading locks are needed for the asyncio.Queue (it's already coroutine-safe)
- `asyncio.sleep(T_optimal)` yields control to the HTTP server to handle gradient submissions during the wait

This is the fundamental reason the hybrid architecture works: `await asyncio.sleep(T)` is NOT a blocking sleep — it surrenders control to the event loop, which continues processing incoming HTTP requests.

---

### Chapter 5: The Reward Function — Balancing Three Objectives

The reward signal is what guides the DRL agent's learning. A badly designed reward produces bad behaviour even with a perfect algorithm.

#### 5.1 The Three-Term Reward

```
R = α · ΔAccuracy  −  β · T_optimal  −  γ · n_dropped
```

**Term 1: α · ΔAccuracy (Model Improvement)**
- ΔAccuracy = new_accuracy − old_accuracy
- Positive if the round improved the model, negative if it regressed
- α = 1.0 (default) — the primary objective
- This term drives the agent toward T values that collect enough gradients to make meaningful progress

**Term 2: −β · T_optimal (Latency Penalty)**
- T_optimal is the raw wait time in seconds
- β = 0.05 (default) — mild penalty
- This prevents the agent from always choosing T=30s (maximum). Longer windows cost real-time latency even if they collect more gradients
- Without this term, the agent would converge to always waiting the maximum allowed time

**Term 3: −γ · n_dropped (Fairness/Inclusion Penalty)**
- n_dropped = number of gradient packets hard-dropped due to staleness > S_max
- γ = 0.2 (default)
- This is the "fairness" term — every dropped slow-device gradient costs the agent
- Without this term, the agent could learn to set T very short, excluding all slow devices (phones), maximising speed at the cost of fairness

#### 5.2 Reward Function Dynamics

| Scenario | R result | Agent learns |
|---|---|---|
| Short T, fast workers only, big acc gain | High positive | Reinforce short T when fast workers dominate |
| Short T, phone gradient dropped, small acc gain | Near zero or negative | Try longer T |
| Long T, all workers included, big acc gain | Large positive minus β·T | Balance — optimal T found |
| Long T, model regresses (stale grads) | Large negative | Shorten T or increase S_max |

#### 5.3 Tuning the Coefficients

**β/γ ratio is the key design choice:**

```
If β >> γ: agent optimises for speed, excluding slow devices
If γ >> β: agent always waits for slowest device, slow overall training
If β ≈ γ/4: balanced — agent avoids dropped gradients but still respects time cost
```

A practical calibration approach: set `γ = 0.2` and `β = 0.05`. Then observe `staleness_ratio` from the `/status` endpoint. If it's consistently above 0.3 (30% dropped), increase γ. If training rounds take too long (T_optimal consistently near T_MAX), increase β.

---

## PART 2 — CODEBASE REFERENCE

---

### Chapter 6: Repository Layout

```
distributed-dl/
│
├── configs/                    ← YAML experiment definitions
│   ├── cnn_mnist.yaml
│   ├── mlp_mnist.yaml
│   ├── rnn_mnist.yaml
│   ├── transformer_mnist.yaml
│   └── nn_mnist.yaml
│
├── models/                     ← Model architecture registry
│   ├── __init__.py             ← Registry: get_model(), register decorator
│   ├── cnn.py                  ← Convolutional Neural Network
│   ├── mlp.py                  ← Multi-Layer Perceptron
│   ├── rnn.py                  ← LSTM Recurrent Network
│   ├── transformer.py          ← Vision Transformer (patch-based)
│   └── neural_net.py           ← Vanilla feedforward baseline
│
├── datasets/                   ← Dataset registry
│   ├── __init__.py             ← Registry: get_dataset(), register decorator
│   └── mnist.py                ← MNIST in-memory tensor wrapper
│
├── common/                     ← Shared utilities
│   ├── config.py               ← YAML loader + defaults
│   └── serialization.py        ← Base64 tensor encode/decode
│
├── controller/                 ← Parameter server
│   ├── __init__.py             ← Package exports
│   ├── controller.py           ← FastAPI app + HTTP endpoints + lifespan
│   ├── scheduler.py            ← Worker hardware registry
│   ├── staleness_filter.py     ← GradientPacket + StalenessFilter
│   ├── sac_orchestrator.py     ← SAC Actor-Critic (T_optimal policy)
│   └── hybrid_controller.py    ← 8-step async training round loop
│
├── worker/
│   ├── trainer.py              ← Local gradient computation engine
│   └── worker.py               ← WorkerNode: fetch → train → send loop
│
├── docs/
│   ├── ARCHITECTURE.md         ← System architecture reference
│   ├── TEXTBOOK_PART1.md       ← This file (theory + codebase reference)
│   └── TEXTBOOK_PART2.md       ← Complete annotated source listings
│
├── normal_train.py             ← Single-machine baseline trainer
└── requirements.txt
```

#### Dependency Layers (bottom → top)

```
Layer 0:  configs/*.yaml         (pure data, no code)
Layer 1:  common/               (no project imports)
Layer 2:  models/, datasets/    (imports common only)
Layer 3:  controller/scheduler  (imports common only)
Layer 4:  controller/staleness_filter  (imports torch only)
Layer 5:  controller/sac_orchestrator (imports torch + staleness_filter)
Layer 6:  controller/hybrid_controller (imports all controller modules)
Layer 7:  controller/controller.py    (imports everything, exposes HTTP)
Layer 8:  worker/trainer.py     (imports common + models)
Layer 9:  worker/worker.py      (imports common + trainer, HTTP calls)
```

---

### Chapter 7: Configuration System

#### 7.1 YAML Schema

Every experiment is defined in one YAML file:

```yaml
# configs/cnn_mnist.yaml

model:
  name: cnn                     # key into models registry
  params:
    in_channels: 1
    num_classes: 10
    input_size: 28
    conv_channels: [32, 64]
    fc_dim: 128
    dropout: 0.25

dataset:
  name: mnist                   # key into datasets registry
  data_dir: ./data

training:
  learning_rate: 0.001          # Adam lr for global model
  batch_size: 64                # samples per gradient step
  target_epochs: 5              # training budget
  max_staleness: 5              # S_max hard-drop threshold
  steps_per_task: 15            # gradient steps per worker task
```

The `model.name` and `dataset.name` strings are looked up in their respective registries at runtime. Changing the YAML is all that's needed to switch from CNN to Transformer — no code changes required.

#### 7.2 `common/config.py` — load_config()

```python
def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Fill in missing training keys with defaults
    training = config.get("training", {})
    for key, default in defaults.items():
        training.setdefault(key, default)
    config["training"] = training
    return config
```

`setdefault` only sets a key if it's absent — existing YAML values always win. Module-level constants (LEARNING_RATE = 0.001 etc.) serve as the default fallback values.

---

### Chapter 8: Model Registry

#### 8.1 The Registration Pattern

```python
# models/__init__.py
_REGISTRY: dict[str, tuple[type[nn.Module], str]] = {}

def register(name: str, description: str = ""):
    def decorator(cls: type[nn.Module]):
        _REGISTRY[name] = (cls, description)
        return cls
    return decorator
```

`@register("cnn")` is a **decorator factory** — `register("cnn")` returns a decorator, which is then applied to the CNN class. This is equivalent to:

```python
class CNN(nn.Module): ...
CNN = register("cnn")(CNN)   # registers and returns the class unchanged
```

The class is stored in `_REGISTRY["cnn"]` and returned unchanged. The class still works normally — the decorator only has the side effect of registration.

#### 8.2 Self-Registration via Import

```python
# models/__init__.py — last line
from . import cnn, mlp, rnn, transformer, neural_net  # noqa
```

Importing a module executes its top-level code. When `cnn.py` is imported, the `@register("cnn")` decorator runs. All models are registered by the time `get_model()` is called — with zero boilerplate in the caller.

#### 8.3 get_model()

```python
def get_model(name: str, params: dict | None = None) -> nn.Module:
    cls, _ = _REGISTRY[name]
    return cls(**(params or {}))    # unpack YAML params as kwargs
```

`**params` unpacks the YAML dict as keyword arguments. `{"fc_dim": 128, "dropout": 0.25}` becomes `CNN(fc_dim=128, dropout=0.25)`.

#### 8.4 Available Architectures

| Registry Key | Class | Input Shape | Notes |
|---|---|---|---|
| `cnn` | CNN | [B, 1, 28, 28] | Conv+BN+ReLU+MaxPool stack, configurable channels |
| `mlp` | MLP | [B, 1, 28, 28] | Flattened to [B, 784], Linear+BN+ReLU+Dropout |
| `rnn` | RNN | [B, 1, 28, 28] | Squeezed to [B, 28, 28], treats rows as LSTM timesteps |
| `transformer` | TransformerClassifier | [B, 1, 28, 28] | 7×7 patches → CLS token → encoder |
| `neural_net` | NeuralNet | [B, 1, 28, 28] | Vanilla FC, no BN/dropout, baseline only |

---

### Chapter 9: The CNN Architecture in Detail

```
Input: [B, 1, 28, 28]
  ↓
Conv2d(1→32, k=3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2)
  Output: [B, 32, 14, 14]
  ↓
Conv2d(32→64, k=3, pad=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2)
  Output: [B, 64, 7, 7]
  ↓
Flatten: [B, 64×7×7] = [B, 3136]
  ↓
Dropout(0.25) → Linear(3136, 128) → ReLU → Dropout(0.25) → Linear(128, 10)
  Output: [B, 10]  ← logits, one per class
```

**BatchNorm2d**: normalises each feature map across the batch, accelerating training and acting as implicit regularisation.

**MaxPool2d(2)**: halves spatial dimensions (28→14→7). Each MaxPool takes the maximum value in each 2×2 window.

**The flat_dim calculation** in code:
```python
feat_size = input_size  # 28
for _ in conv_channels:
    feat_size = feat_size // 2  # 28 → 14 → 7
flat_dim = conv_channels[-1] * feat_size * feat_size  # 64 × 7 × 7 = 3136
```

---

### Chapter 10: The Vision Transformer in Detail

The Transformer architecture treats an image as a **sequence of patches**, similar to how NLP Transformers treat text as a sequence of tokens.

#### 10.1 Patchification

For a 28×28 image with patch_size=7:
```
num_patches = (28/7)² = 16
patch_dim = 1 × 7 × 7 = 49  (channels × height × width)
```

Each patch is flattened and linearly projected to `d_model=128`.

#### 10.2 CLS Token

A learnable `[CLS]` token is prepended to the sequence:
```
[CLS, patch_0, patch_1, ..., patch_15]  → sequence length = 17
```

After the Transformer encoder, only the CLS token's representation is used for classification. This is the ViT (Vision Transformer) design from Dosovitskiy et al. 2020.

#### 10.3 Positional Encoding

Pure attention has no notion of order — patch 0 and patch 15 look the same without position information. Learned positional embeddings (one per position) are added to each token before the encoder.

#### 10.4 Multi-Head Self-Attention

For each token, attention computes how much it should "attend to" every other token:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
Multiple "heads" run in parallel, each learning different relationship patterns.
