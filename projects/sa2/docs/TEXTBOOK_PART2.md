# Textbook Part 2 — Annotated Source Walk-Through

## Chapter 11: common/serialization.py

Workers and the controller communicate over HTTP with JSON bodies. PyTorch
tensors cannot be JSON-serialised directly, so every tensor is converted to a
Base64 string before sending and decoded on receipt.

```
Tensor → torch.save(BytesIO) → base64.b64encode → UTF-8 string → JSON
JSON  → base64.b64decode     → BytesIO           → torch.load  → Tensor
```

### Why torch.save/load instead of numpy tolist()?

- `torch.save` preserves dtype (float32, int64, etc.) and device metadata exactly.
- `tolist()` would convert to Python floats, losing dtype and requiring
  explicit reconstruction on the other side.
- For large gradient tensors (millions of floats), BytesIO + base64 is compact
  and round-trip exact.

### Key functions

```python
encode_tensor(t)        # Tensor → str
decode_tensor(s)        # str → Tensor
encode_state_dict(sd)   # dict[str, Tensor] → str
decode_state_dict(s)    # str → dict[str, Tensor]
encode_gradients(grads) # list[Tensor] → str
decode_gradients(s)     # str → list[Tensor]
```

`weights_only=True` in `torch.load` is a security flag — it prevents arbitrary
Python object deserialisation (pickle exploit protection).

---

## Chapter 12: datasets/mnist.py — MNISTDataset

### Pre-loading Strategy

```python
self._train_data = torch.stack([img for img, _ in train_ds])
```

All 60,000 training images are loaded into RAM as a single tensor at startup.
This trades memory (~180 MB) for speed — every `get_batch()` call is a tensor
slice, not a disk read.

### Thread-Safe get_batch()

```python
def get_batch(self, size: int):
    with self._lock:           # threading.Lock — prevents race conditions
        ...
        while remaining > 0:
            # slice from current position
            # if we reach end, reshuffle and wrap around
```

The lock matters because multiple HTTP requests can call `get_batch()` concurrently
(FastAPI handles requests in a thread pool for sync functions). Without the lock,
two requests could both read `self._data_idx`, both advance it, and return
overlapping or skipped batches.

**Wrap-around reshuffle**: when the pointer reaches the end of 60,000 samples,
the data is randomly permuted and the pointer resets to 0. This approximates
epoch-based training without requiring explicit epoch tracking in the caller.

---

## Chapter 13: worker/trainer.py — Trainer

The Trainer is intentionally simple: it receives a model (any architecture)
and runs local gradient accumulation.

### Key Design: Gradient Accumulation Without optimizer.step()

```python
self.model.zero_grad()
for step in range(steps):
    outputs = self.model(batch)
    loss = self.criterion(outputs, labels)
    loss.backward()   # accumulates .grad on each parameter
    valid_steps += 1

# Normalise: average gradients across all steps
if valid_steps > 1:
    for param in self.model.parameters():
        if param.grad is not None:
            param.grad /= valid_steps
```

`loss.backward()` **adds** to `.grad` each call (PyTorch accumulates by default).
After all steps, dividing by `valid_steps` gives the mean gradient across the
entire task — equivalent to training on all batches at once with a single update.

The worker **never calls `optimizer.step()`**. It only computes gradients and
sends the `.grad` tensors. The controller's optimizer performs the actual weight
update.

### Why No Optimizer on the Worker?

Workers are stateless. Each task starts from a fresh model copy sent by the
controller. If workers had optimizers, their momentum states would be stale and
inconsistent with the server's optimizer state. Gradient-only submission keeps
everything consistent — the controller's Adam optimizer maintains the canonical
momentum buffers.

---

## Chapter 14: worker/worker.py — WorkerNode

### The Dual-Thread Architecture

```
Main Thread                         Background Daemon Thread
──────────────────────────────      ─────────────────────────────────
task = task_queue.get()  ◄──────── fetch_task_loop():
                                      GET /get_task
train locally                         task_queue.put(task)
send gradients (daemon thread)
next task_queue.get() [may block]
```

**Why a separate fetch thread?**

Without it, the sequence would be: train → wait for send → fetch next task →
train. The network round-trip for fetching (potentially 100ms–2s) would stall
local computation. With the daemon thread, while the main thread is training,
the background thread is already fetching the next task. When training finishes,
the next task is (usually) already in the queue.

### Fire-and-Forget Gradient Sending

```python
def send_async(payload):
    requests.post(f"{url}/send_gradients", json=payload)

threading.Thread(target=send_async, args=(payload,), daemon=True).start()
```

Gradient sending is dispatched to yet another daemon thread. The main loop
moves immediately to `task_queue.get()` — it does not wait for the HTTP
response confirming gradient receipt. This overlaps three operations:

```
Time ──────────────────────────────────────────────────────►
  Main:    [train task N] [train task N+1] [train task N+2]
  Fetch:         [fetch N+1]   [fetch N+2]
  Send:      [send N]   [send N+1]   [send N+2]
```

### Bandwidth Measurement

```python
download_time = time.time() - req_start_time
payload_size  = len(r.content)
self.bandwidth = payload_size / max(download_time, 0.001)
```

Each `get_task` response carries the model state dict (potentially several MB).
By timing this download and dividing by bytes received, the worker gets a
real-time bandwidth estimate (bytes/sec). This is reported back to the controller
on the next `get_task` request and becomes an input to the DRL state vector.

---

## Chapter 15: controller/scheduler.py — Scheduler

```python
class WorkerStats:
    worker_id         : str
    cpu_cores         : int
    ram               : int       # MB
    device_type       : str       # "phone", "laptop", "unknown"
    samples_per_second: float     # updated by worker on each /get_task
    bandwidth         : float     # bytes/sec, updated same
```

The Scheduler is a simple dictionary registry. The `samples_per_second` and
`bandwidth` fields are updated every time a worker calls `/get_task` with
its measured stats from the previous task. These live measurements form the
DRL's view of the current network state.

`compute_optimal_steps()` currently returns a fixed value from config. This
is where future work could dynamically size each worker's task based on its
compute speed to make all workers finish in roughly the same time.

---

## Chapter 16: controller/staleness_filter.py — Full Walk-Through

### GradientPacket

```python
@dataclass
class GradientPacket:
    worker_id         : str
    gradients         : list[torch.Tensor]  # one per model parameter
    worker_version    : int                 # version when worker fetched model
    samples_processed : int                 # how much data was used
    received_at       : float               # auto-set to time.time()
```

`received_at` is not currently used in filtering decisions but is available
for future latency analysis (e.g., flagging packets that sat in the queue
for unexpectedly long).

### StalenessFilter.evaluate()

```python
staleness = global_version - packet.worker_version
staleness = max(staleness, 0)          # guard: version can't go backwards

if staleness > self.max_staleness:
    return FilterResult(..., weight=0.0, is_valid=False)

weight = 1.0 - (staleness / self.max_staleness)
weight = max(weight, 0.01)             # floor: never exactly 0 for valid grads
return FilterResult(..., weight=weight, is_valid=True)
```

The `max(staleness, 0)` guard handles a rare edge case: if a worker fetched
the model AFTER the current round started (worker_version > global_version at
receipt time), staleness would be negative. Clamping to 0 treats it as fresh.

### StalenessFilter.weighted_average()

```python
eff_weights = [r.weight * r.packet.samples_processed for r in valid_results]
total_weight = sum(eff_weights) or 1.0

agg = [torch.zeros_like(valid_results[0].packet.gradients[i]) ...]

for result, ew in zip(valid_results, eff_weights):
    for i, grad in enumerate(result.packet.gradients):
        agg[i] += (ew / total_weight) * grad
```

This is a weighted federated average. The `/ total_weight` normalises so
coefficients sum to 1.0 — mathematically correct gradient averaging.
`or 1.0` prevents division by zero if all effective weights are 0.

---

## Chapter 17: controller/sac_orchestrator.py — Full Walk-Through

### The _MLP Backbone

```python
class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
```

**Why Tanh instead of ReLU?**
For RL networks, Tanh is preferred over ReLU because:
1. Tanh outputs in [-1, 1] — bounded activations help with stable value estimation
2. ReLU "dying" (neurons stuck at 0) can cause RL instability
3. The bounded output range works better with the small state vector magnitudes

### Actor Forward Pass

```python
features  = self.backbone(state)          # (1, 64)
raw_mean  = self.mean_head(features)      # (1, 1) — unconstrained
mean      = sigmoid(raw_mean) * (T_MAX - T_MIN) + T_MIN  # (1, 1) in [0.5, 30]
log_std   = self.log_std_head(features).clamp(-4.0, 1.0)
```

**log_std clamping [-4, 1]:**
- exp(-4) ≈ 0.018 seconds std — very deterministic (exploitation)
- exp(1)  ≈ 2.72 seconds std  — significant exploration spread
- Prevents numerical explosion from very large/small σ values

### sample_action() — Reparameterisation Trick

```python
raw = dist.rsample()             # μ + σ·ε where ε ~ N(0,1)  [differentiable]
action = raw.clamp(T_MIN, T_MAX) # safety clip for the sampled value
log_prob = dist.log_prob(raw).sum(dim=-1)
```

`.rsample()` (reparameterised sample) is differentiable through the sampling
operation. This is crucial for gradient-based policy optimisation.

Note: `log_prob` is computed on `raw` (before clamp), not on `action`. This
avoids incorrect probability density at the boundary values.

### SAC Update — Step by Step

```python
# 1. Probability ratio
ratio = torch.exp(new_log_prob - old_log_prob)
#    = π_new(a|s) / π_old(a|s)   [in log space for numerical stability]

# 2. Clipped surrogate
surr1 = ratio * advantages
surr2 = ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * advantages
actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_C * entropy

# 3. Gradient clipping for stability
nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
```

**Why -torch.min(surr1, surr2)?**
We want to MAXIMISE the clipped surrogate. Calling `.backward()` minimises by
default, so we negate it. `min(surr1, surr2)` is the pessimistic (conservative)
estimate — we take the smaller of the clipped and unclipped objectives,
preventing the policy from over-updating.

**Gradient clipping (max_norm=0.5):**
Rescales the gradient vector if its L2 norm exceeds 0.5. Prevents catastrophic
parameter updates that could destabilise the policy. Standard practice in RL.

### _compute_returns() — Monte Carlo Returns

```python
running = 0.0
for t in reversed(range(T)):
    running = rewards[t] + GAMMA * running * (0.0 if dones[t] else 1.0)
    returns[t] = running
```

Processes rewards backward in time. Each G_t = r_t + γ·G_{t+1}.
`dones[t]` flags episode termination — if True, future rewards don't carry
over (currently always False in this framework since training is continuous).

---

## Chapter 18: controller/hybrid_controller.py — The Core Loop

### Constructor — Dependency Injection

```python
HybridController(
    model, optimizer, scheduler,
    evaluate_fn,      # async callable → float
    gradient_queue,   # asyncio.Queue (shared with HTTP layer)
    max_staleness, device, target_samples, dataset_size
)
```

Every dependency is injected — the HybridController does not import or create
any global state. This makes it testable and avoids circular imports.

### The 8-Step Round in Code

**Step 1-2: Observe and Act**
```python
state = self.drl.build_state(
    scheduler, self.last_accuracy, self.filter.staleness_ratio)
t_optimal = self.drl.act(state)
```

**Step 3: Hybrid Wait**
```python
await asyncio.sleep(t_optimal)
# During this await, the event loop handles /send_gradients requests
# Each request calls gradient_queue.put(packet) — completely non-blocking
```

**Step 4: Drain (Sync Cutoff)**
```python
def _drain_queue(self) -> list[GradientPacket]:
    packets = []
    while True:
        try:
            packet = self.gradient_queue.get_nowait()
            packets.append(packet)
        except asyncio.QueueEmpty:
            break
    return packets
```

`get_nowait()` is the key: it does NOT await. It either returns an item
immediately or raises QueueEmpty. The while loop drains every available
packet in O(n) time with zero blocking.

**Step 5-6: Filter and Aggregate**
```python
valid, dropped = self.filter.filter_batch(packets, self.global_version)
agg_grads = StalenessFilter.weighted_average(valid)

if agg_grads is not None:
    async with self._grad_lock:
        await self._apply_gradients(agg_grads, valid)
```

The `async with self._grad_lock` ensures the optimizer step is atomic —
no other coroutine can modify model weights during the update.

**Step 7-8: Evaluate, Reward, SAC Update**
```python
new_accuracy = await self.evaluate_fn()
acc_improvement = new_accuracy - self.last_accuracy

reward = ALPHA * acc_improvement - BETA * t_optimal - GAMMA_PENALTY * n_dropped

self.drl.store_reward(reward)
drl_metrics = self.drl.update()   # SAC step

self.global_version += 1
```

---

## Chapter 19: controller/controller.py — HTTP Layer

### Lifespan Hook (Startup/Shutdown)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hybrid_ctrl
    _hybrid_ctrl = HybridController(...)
    loop_task = asyncio.create_task(
        _hybrid_ctrl.run_training_loop(),
        name="hybrid-training-loop"
    )
    yield           # server handles requests here
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass        # clean shutdown
```

`asyncio.create_task()` schedules the coroutine and returns immediately.
The `yield` marks where the application runs. On shutdown (Ctrl+C or SIGTERM),
the code after `yield` executes — `loop_task.cancel()` sends a
CancelledError into the training loop at its next await point.

### /send_gradients — Intentionally Thin

```python
@app.post("/send_gradients")
async def send_gradients(req: SendGradientsRequest):
    grads  = decode_gradients(req.gradients)
    packet = GradientPacket(
        worker_id=req.worker_id,
        gradients=grads,
        worker_version=req.model_version,
        samples_processed=req.samples_processed,
    )
    await gradient_queue.put(packet)    # O(1), never blocks
    return {"status": "queued", ...}
```

No staleness check. No model update. No lock. Just decode and enqueue.
This endpoint handles potentially hundreds of concurrent requests from all
workers — keeping it thin is critical for throughput.

### /get_task — Pull-Based Distribution

Workers pull work on demand, not pushed by the controller. This naturally
handles variable worker speeds: a fast worker simply calls /get_task more often.
The controller doesn't need to know how fast workers are to keep them busy.

```python
@app.get("/get_task")
async def get_task(worker_id, samples_per_sec=0.0, bandwidth=0.0):
    # Update scheduler with worker's measured performance
    scheduler.update_worker_stats(worker_id, samples_per_sec, bandwidth)

    data, labels = dataset.get_batch(k * BATCH_SIZE)

    return {
        "model_state": encode_state_dict(model.state_dict()),
        "model_version": ctrl.current_version,   # ← critical for staleness calc
        ...
    }
```

The returned `model_version` is the global_version at the moment of the request.
When the worker later submits gradients, this version number travels back and
lets the staleness filter compute exactly how many rounds the model has moved
since this gradient was computed.

---

## Chapter 20: End-to-End Data Flow Narrative

Here is the complete life of one piece of data, from dataset to model update:

**1. Dataset (controller startup)**
MNIST loads 60,000 images into `_train_data` tensor. Pointer starts at 0.

**2. Round start (HybridController)**
DRL observes state, outputs T_optimal = 6.2s. global_version = 47.

**3. Worker fetch (/get_task)**
Worker calls GET /get_task. Controller slices 960 samples (15 steps × 64
batch), serialises model (version=47), returns task. Worker's
model_version=47 is stored locally.

**4. Local training (worker/trainer.py)**
Worker runs 15 forward+backward passes, accumulating gradients. Takes ~3.1s.
Returns list of gradient tensors (one per model parameter).

**5. Gradient submission (/send_gradients)**
Worker calls POST /send_gradients with:
- gradients: base64(list of Tensors)
- model_version: 47
- samples_processed: 960

Controller decodes, wraps in GradientPacket(worker_version=47), puts in queue.
Returns {"status": "queued"} immediately.

**6. T_optimal expires (HybridController)**
asyncio.sleep(6.2) returns. _drain_queue() sweeps all packets.

**7. Staleness filter**
global_version is now 48 (incremented at end of last round).
staleness = 48 - 47 = 1.
weight = 1.0 - 1/5 = 0.80.

**8. Weighted aggregate**
effective_weight = 0.80 × 960 = 768.
Gradient scaled by 768/total_weight and added to aggregate.

**9. Model update**
optimizer.zero_grad() → param.grad = agg_grad → optimizer.step().
global_version becomes 49.

**10. Evaluation**
_evaluate_model_sync() runs on test set in thread pool. Returns 83.4%.

**11. Reward**
R = 1.0 × (83.4 - 82.1) - 0.05 × 6.2 - 0.2 × 0 = 1.3 - 0.31 = 0.99

**12. SAC update**
DRL stores reward 0.99. update() runs 4 SAC epochs on accumulated trajectory.
Actor learns: "6.2s was a good choice in this kind of state."

---

## Chapter 21: Failure Modes and Mitigations

| Failure | Symptom | Mitigation in Code |
|---|---|---|
| Worker crashes mid-task | Gradients never arrive | Timeout: they simply miss the window, counted as 0 dropped |
| Worker very slow (phone) | high staleness_ratio | γ penalty forces DRL to widen T window |
| Network partition | No gradients arrive | `if not packets: penalise and continue` — training round still advances |
| Model divergence from stale grads | Accuracy decreases | Negative ΔAcc → large negative reward → DRL tightens S_max indirectly via T |
| Controller restart | global_version resets | Workers auto-re-register; old gradients get staleness = version_now - 0 = large → dropped |
| Queue overflow | Memory growth | asyncio.Queue has no maxsize limit; monitor via /status queue_size |

---

## Chapter 22: How to Run the System

### Prerequisites
```bash
pip install fastapi uvicorn torch torchvision requests tqdm psutil pyyaml
```

### Start Controller
```bash
cd distributed-dl
CONFIG_PATH=configs/cnn_mnist.yaml uvicorn controller.controller:app \
    --host 0.0.0.0 --port 8000
```

### Start Workers (any number, any device on same network)
```bash
# On the same machine:
python worker/worker.py --controller-ip 127.0.0.1

# On another device:
python worker/worker.py --controller-ip <CONTROLLER_LAN_IP>
```

### Monitor Training
```bash
curl http://localhost:8000/status
# {"round": 14, "global_version": 14, "total_samples": 53760,
#  "accuracy": 84.2, "staleness_ratio": 0.08, "queue_size": 0}
```

### Single-Machine Baseline
```bash
python normal_train.py --config configs/cnn_mnist.yaml
python normal_train.py --list-models
```

### Add a New Model Architecture
1. Create `models/your_model.py` with `@register("your_model")` decorator
2. Create `configs/your_model_mnist.yaml` with `model.name: your_model`
3. No other changes needed — both controller and workers auto-discover it

---

## Chapter 23: Extension Points

| Feature | Where to Add | Notes |
|---|---|---|
| New dataset | `datasets/new.py` with `@register` | Implement `get_batch()` and `get_test_loader()` |
| New model | `models/new.py` with `@register` | Must accept kwargs matching YAML `params` |
| Richer DRL state | `sac_orchestrator.py::build_state()` | Increase `STATE_DIM`, add new features |
| Different RL algo | Replace `sac_orchestrator.py` | Keep same `act()` / `store_reward()` / `update()` interface |
| Gradient compression | `common/serialization.py` | Quantise or sparsify tensors before encode |
| Secure comms | `controller.py` | Add FastAPI dependency for auth token on each request |
| Multi-controller | New routing layer | Workers choose controller by latency or data shard |
