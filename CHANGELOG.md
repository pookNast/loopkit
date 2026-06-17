# Changelog

## 0.2.0

### Ensemble (Gem 7)
- **Voter independence.** `EnsembleEvaluator` now accepts a `voters=[...]` list of
  distinct callables so each vote comes from an independent error mode (different
  model / temperature / seed / prompt). The Condorcet accuracy lift only
  materializes when voters actually differ; calling one deterministic `cheap_fn`
  three times produces identical votes and majority voting adds nothing.
- **Tie-only escalation.** The expensive tiebreaker now fires only when no verdict
  holds a strict majority (default `escalate_on="no_majority"`), not on every 2-1
  split. Pass `escalate_on="any_disagreement"` for the legacy behaviour.
- `condorcet_accuracy()` validates `0 <= p <= 1` and `n_voters >= 1`.

### CUSUM (Gem 2)
- **Auto-calibration is now opt-in.** `CUSUMBank.register(..., baseline=...)` uses
  the explicit baseline as-is by default (`auto_calibrate=False`). Previously the
  default silently replaced a caller-supplied baseline with the running mean.
  Pass `auto_calibrate=True` to keep the calibration-window behaviour.

### Beliefs (Gem 1)
- `BeliefEngine.select_best(..., min_obs=N)` now enforces a forced-exploration
  warmup: until every candidate has `min_obs` observations, the least-tried
  candidate is returned instead of sampling.

### Persistence (store)
- `SQLiteStore` is thread-safe (`check_same_thread=False` + internal lock),
- usable as a context manager, and exposes `save_all(...)` for an atomic
  multi-gem snapshot (single transaction, rollback on error).
- `Orchestrator.save()` now persists via `save_all`.

### Packaging
- Added `py.typed` marker, ruff config, `[dev]` extra, and CI workflow
  (ruff + pytest on Python 3.10 / 3.11 / 3.12 / 3.13).

## 0.1.0
- Initial release: seven composable gems.
