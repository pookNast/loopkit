# loopkit

**Probabilistic orchestration primitives for AI agent loops. Zero dependencies.**

loopkit gives your agent loops a brain. Instead of hardcoded routing tables, fixed retry counts, and threshold-based alerts, loopkit uses Bayesian beliefs, statistical anomaly detection, and information-theoretic spin detection -- all in Python stdlib with no external dependencies.

```
pip install loopkit
```

## Why

> "Here's your monthly reminder that you shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents." -- @steipete

The loop is plumbing. The interesting engineering is everything you wrap around the loop's decisions so it doesn't run off a cliff.  loopkit provides seven composable primitives ("gems") for that wrapping:

| # | Gem | What it replaces |
|---|-----|-----------------|
| 1 | **Beta Beliefs** | Hardcoded model routing tables |
| 2 | **CUSUM** | Threshold-based alerting |
| 3 | **NCD Spin Detection** | Fixed retry caps |
| 4 | **Circuit Breaker** | Binary health checks |
| 5 | **Skill DAG** | Blind chain execution |
| 6 | **Degradation Ladder** | All-or-nothing failure |
| 7 | **Ensemble Evaluator** | Single expensive evaluator |

**Architecture: Probabilistic Core, Deterministic Shell.** The probabilistic layer influences routing, retry timing, and alerting.  It never overrides hard stops (max iterations, budget ceiling, concurrency limits).

## Quick Start

```python
from loopkit import Orchestrator

# Create an orchestrator (optional: persist to SQLite)
ork = Orchestrator(db_path="loopkit.db")

# Record outcomes -- beliefs update automatically
ork.record_outcome("model", "sonnet", success=True)
ork.record_outcome("model", "opus", success=True)
ork.record_outcome("model", "haiku", success=False)

# Thompson Sampling picks the best model
model = ork.select_model(["haiku", "sonnet", "opus"], job_type="debug")

# Check if a host is healthy (circuit breaker)
decision = ork.should_dispatch(job_priority=20, host="sandbox")
if not decision.allowed:
    print(f"Blocked: {decision.reason}")

# Detect spinning loops
for output in agent_outputs:
    if ork.check_spin("job-123", output):
        print("Loop is spinning -- halting")
        break

# Full health report
print(ork.health_report())
```

## The Seven Gems

### 1. Beta Beliefs + Thompson Sampling

Every entity gets a `Beta(alpha, beta)` prior that updates on every outcome.  Thompson Sampling selects the best option by sampling from posteriors -- no hardcoded routing tables needed.

```python
from loopkit import BeliefEngine

engine = BeliefEngine(decay_factor=0.99)

# Record outcomes
engine.update("model", "opus", success=True, context="debug")
engine.update("model", "sonnet", success=True, context="debug")
engine.update("model", "sonnet", success=False, context="debug")

# Thompson Sampling: who's best for debug tasks?
best = engine.select_best("model", ["opus", "sonnet", "haiku"], context="debug")

# Credible intervals
belief = engine.get("model", "opus", "debug")
print(f"P(success) = {belief.mean:.2f}, 95% CI = {belief.credible_interval()}")

# Active learning: which entity has highest uncertainty?
ranked = engine.ranked("model", by="variance")  # test this one next
```

**Math:**
```
Prior:    θ ~ Beta(α, β)
Update:   α += success, β += failure  
Sample:   θ̂ = Gamma(α,1) / (Gamma(α,1) + Gamma(β,1))
CI:       Normal approximation when α+β > 10
Decay:    α *= λ, β *= λ  (recency weighting, λ=0.99/day)
```

### 2. CUSUM Anomaly Detection

Catches *slow drift* that threshold alerts miss.  A success rate dropping from 85% to 70% over 3 days fires CUSUM long before any single-point threshold.

```python
from loopkit import CUSUMBank

bank = CUSUMBank()
bank.register("success_rate", baseline=0.85, allowance_k=0.05, threshold_h=4.0)
bank.register("avg_cost", baseline=1.50, allowance_k=0.10, threshold_h=5.0)
# baselines are used as-is. Pass auto_calibrate=True to instead derive the
# baseline from the mean of the first calibration_window (30) observations.

# Feed observations each cycle
alerts = bank.update_many({
    "success_rate": current_success_rate,
    "avg_cost": current_avg_cost,
})

# Early warning: which metrics are approaching threshold?
warnings = bank.alerts()  # metrics > 80% toward threshold
```

**Math:**
```
S⁺ₙ = max(0, S⁺ₙ₋₁ + (xₙ - μ₀ - K))   # upper CUSUM
S⁻ₙ = max(0, S⁻ₙ₋₁ - (xₙ - μ₀ + K))   # lower CUSUM
Alert when S⁺ₙ > H or S⁻ₙ > H
```

### 3. NCD Spin Detection

Uses gzip compression to measure information-theoretic similarity between consecutive loop outputs.  Catches agents paraphrasing the same wrong fix.

```python
from loopkit import SpinDetector, ncd

# Raw NCD between two strings
similarity = ncd("attempt 1 output", "attempt 2 output")
# 0.0 = identical, 1.0 = maximally different

# Automatic spin detection
detector = SpinDetector(epsilon=0.15, window=3)
for output in loop_outputs:
    if detector.feed(output):
        print(f"Spinning detected! Mean NCD = {detector.mean_ncd:.3f}")
        break
```

**Math:**
```
NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
where C(·) = len(gzip.compress(·))
Spin when mean(NCD) over window < ε
```

### 4. Circuit Breaker

Three-state FSM with Bayesian adaptive cooldown.  Unreliable services get longer cooldowns automatically.

```python
from loopkit import CircuitBreakerRegistry

breakers = CircuitBreakerRegistry(default_threshold=3)

if breakers.should_attempt("sandbox"):
    try:
        result = call_sandbox(task)
        breakers.record("sandbox", success=True)
    except ConnectionError:
        breakers.record("sandbox", success=False)
else:
    print(f"Sandbox circuit open, cooldown remaining")

# Which services are down?
print(breakers.open_services())
```

**States:** `CLOSED → OPEN → HALF_OPEN → CLOSED`  
**Cooldown:** `base_cooldown / max(P(healthy), 0.01)` -- adapts from Beta posterior.

### 5. Skill DAG + Chain Reliability

Model skill chains as a graph.  Compute calibrated chain reliability with confidence intervals and automatic bottleneck identification.

```python
from loopkit import SkillDAG, BeliefEngine

beliefs = BeliefEngine()
dag = SkillDAG(beliefs)
dag.add_chain(["build", "test", "deploy", "verify"])

# After collecting data...
rel = dag.chain_reliability("build")
print(f"Chain: {rel.chain}")
print(f"Reliability: {rel.point_estimate:.1%} [{rel.ci_lower:.1%}, {rel.ci_upper:.1%}]")
print(f"Bottleneck: {rel.bottleneck} ({rel.bottleneck_mean:.1%})")

# Active learning: which skill to invest in next?
print(f"Highest uncertainty: {dag.highest_uncertainty_skill()}")
```

**Math:**
```
P(chain) = ∏ E[θᵢ]  for each skill i in chain
CI via log-normal approximation of product of Betas
Bottleneck = argmin_i E[θᵢ]
```

### 6. Graceful Degradation Ladder

Five-level hierarchy that tells the system what to sacrifice when resources are constrained.

```python
from loopkit import DegradationLadder, SystemMetrics

ladder = DegradationLadder()

metrics = SystemMetrics(
    budget_remaining_pct=15,
    host_load=12,
    failed_24h=8,
    running=5,
    max_concurrent=6,
)

level = ladder.compute(metrics)
print(f"Level: {level.name}")

# Should this job run?
if ladder.should_dispatch(job_priority=40):
    model = ladder.effective_model("opus")  # may downgrade to haiku
```

**Levels:**
| Level | Name | Action |
|-------|------|--------|
| 0 | NORMAL | All systems go |
| 1 | CONSTRAINED | Pause batch jobs |
| 2 | STRESSED | Reduce concurrency, downgrade models |
| 3 | CRITICAL | Essential only, cheapest models |
| 4 | EMERGENCY | Halt all dispatch, alert human |

### 7. Condorcet Ensemble Evaluation

For binary gates (PASS/FAIL), 3 cheap evaluators with majority vote outperform 1 expensive evaluator -- at 60% of the cost.

```python
from loopkit import EnsembleEvaluator, condorcet_accuracy

# What accuracy does 3-voter majority give?
print(f"3 voters at 70%: {condorcet_accuracy(0.70, 3):.1%}")  # 78.4%
print(f"3 voters at 80%: {condorcet_accuracy(0.80, 3):.1%}")  # 89.6%

evaluator = EnsembleEvaluator(
    cheap_fn=lambda task: run_haiku_eval(task),     # returns "PASS" or "FAIL"
    expensive_fn=lambda task: run_opus_eval(task),   # optional tiebreak
    n_voters=3,
    cheap_cost=0.2,
    expensive_cost=1.0,
)

result = evaluator.evaluate(my_task)
print(f"Verdict: {result.verdict}, agreement: {result.agreement_ratio:.0%}")
print(f"Cost ratio vs single expensive: {result.cost_ratio}")
```

> **Independence matters.** The Condorcet lift only materializes when voters make *independent* errors. Three calls to the same deterministic model (temperature 0, fixed seed) return identical verdicts and majority voting adds nothing. For real diversity pass distinct callables via `voters=[...]` -- different models, temperatures, seeds, or prompts:
> ```python
> evaluator = EnsembleEvaluator(
>     voters=[
>         lambda t: run_model(t, model="haiku", temperature=0.3),
>         lambda t: run_model(t, model="glm-flash", temperature=0.3),
>         lambda t: run_model(t, model="qwopus", temperature=0.3),
>     ],
>     expensive_fn=lambda t: run_model(t, model="opus", temperature=0.0),
> )
> ```
>
> By default the expensive tiebreaker fires **only when no verdict holds a strict majority** (e.g. a 3-way split), not on a normal 2-1 split -- that is where the cost saving comes from. Pass `escalate_on="any_disagreement"` to re-check every non-unanimous vote.

## Persistence

All gems work in-memory by default.  Add SQLite persistence with one argument:

```python
ork = Orchestrator(db_path="loopkit.db")

# ... use normally ...

ork.save()  # persist all state atomically (single transaction)
# State auto-loads on next Orchestrator(db_path="loopkit.db")
```

`SQLiteStore` is thread-safe (a single WAL connection guarded by an internal lock),
usable as a context manager, and exposes `save_all(...)` for an atomic multi-gem
snapshot so a crash mid-save cannot leave half-persisted state:

```python
from loopkit import SQLiteStore

with SQLiteStore("loopkit.db") as store:
    store.save_all(engine=engine, cusum=bank, breakers=registry, dag=dag)
```

Or use `SQLiteStore` directly for fine-grained control:

```python
from loopkit import SQLiteStore, BeliefEngine

store = SQLiteStore("loopkit.db")
engine = BeliefEngine()
# ... use engine ...
store.save_beliefs(engine)
engine = store.load_beliefs()
```

## Design Principles

1. **Zero dependencies.** Python 3.10+ stdlib only.  `gzip`, `random.gammavariate`, `math`, `sqlite3` -- that's the full stack.

2. **Probabilistic core, deterministic shell.** Beliefs and sampling influence *routing decisions*.  Hard stops (max iterations, budget ceilings, concurrency limits) are enforced deterministically and are never overridden by probabilistic components.

3. **Composable.** Use one gem or all seven.  The `Orchestrator` composes them, but each module works standalone.

4. **Observable.** Every component has a `to_dict()` method for JSON-serializable health reports.

## Origin

Born from a homelab orchestration system running 91 skills across multiple hosts with 24/7 autonomous agent loops.  The seven gems were extracted from production patterns and backed by research from arXiv:2605.00742 (Bayes-Consistent Orchestration), arXiv:2604.05333 (Graph-of-Skills), arXiv:2409.00094 (Condorcet + LLMs), and others.

## License

MIT
