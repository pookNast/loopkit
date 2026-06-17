# Milestone Contract: loopkit CLI

Status: **LOCKED** — do not deviate without explicit approval.
Source of architecture: ponytail review (Option A — CLI only).
Reviewer rule: https://github.com/DietrichGebert/ponytail (AGENTS.md).

## Architecture (LOCKED)

- **Shape**: a single `loopkit` console-script (argparse subcommands). Each call
  is a short-lived Python process: open `Orchestrator(db_path)`, do one op,
  `save()` if mutating, print one JSON line to stdout, exit.
- **No daemon, no HTTP, no socket activation.** Decision frequency is ~1 call
  per few seconds; a 50–80 ms process spawn is invisible against agent latency.
- **State split** (the correctness-critical lock):
  - **SQLite-backed (load → op → save)**: `record-outcome`, `select-model`,
    `should-dispatch`, `health`. These touch beliefs / cusum / breakers, which
    `SQLiteStore` already persists.
  - **Stateless per-invocation**: `check-spin` and `eval`. Spin-detector windows
    are NOT persisted by `SQLiteStore`, so these commands take their full input
    each call (recent outputs / task + voters) and run a fresh detector in
    process. No persistence needed, no schema change.
- **Dependencies**: zero new. stdlib only: `argparse`, `json`, `os`, `sys`,
  `subprocess`, `shlex`. Reuse `loopkit.Orchestrator`, `EnsembleEvaluator`,
  `SpinDetector`, `ncd` from the existing package.

## Stack Constraints

- NO `click` / `typer` / `rich` — argparse only.
- NO FastAPI / http.server / uvicorn sidecar.
- NO pydantic-settings / config layer — one env var covers config.
- NO plugin/registry system, NO telemetry, NO `--format table`.
- All changes must pass `ruff check src tests` and `pytest -q`.

## CLI Surface (LOCKED — exact flags)

Global: every subcommand accepts `--db PATH` (default: `$LOOPKIT_DB` or
`~/.loopkit/state.db`). Parent dir is created if missing.

Output contract: **exactly one JSON line on stdout**, nothing else (no logs, no
pretty mode). Exit codes noted per command. Errors go to stderr with exit 2.

Input convention (`TEXT` args for `--output`/`--input`/`--task`): a literal
string, OR `@PATH` to read from a file, OR `-` to read from stdin.

| Subcommand | Flags | Maps to | Exit |
|---|---|---|---|
| `record-outcome` | `--type T --id ID (--success\|--fail) [--context CTX] [--host H]` | `ork.record_outcome(...)` then `ork.save()` | 0 |
| `select-model` | `--candidates a,b,c --job-type T [--min-obs N=5] [--fallback X]` | `ork.select_model(...)` (reads beliefs) | 0 |
| `should-dispatch` | `--priority N --host H` | `ork.should_dispatch(...)` | 0 if allowed, **1 if denied** |
| `health` | (none) | `ork.health_report()` | 0 |
| `check-spin` | `--epsilon E=0.15 --window W=3 --from FILE` (one output per line; omit `--from` → stdin) | fresh `SpinDetector`, feed each line | 0 if not spinning, **1 if spinning** |
| `eval` | `--voter CMD` (repeatable, ≥1) `[--tiebreak CMD] [--input TEXT\|@FILE\|-]` | `EnsembleEvaluator(voters=[subprocess fns], expensive_fn=tiebreak)`, task from input | 0 |

### Output JSON shapes (locked)
- `record-outcome` → `{"status":"ok","entity_type":...,"entity_id":...,"success":bool}`
- `select-model` → `{"model":"sonnet"}`
- `should-dispatch` → `{"allowed":bool,"reason":...,"degradation_level":...,"model_override":...,"circuit_breaker_open":bool}`
- `health` → the full `health_report()` dict
- `check-spin` → `{"spinning":bool,"mean_ncd":float|null,"spin_count":int,"outputs_seen":int}`
- `eval` → `{"verdict":...,"agreement_ratio":float,"votes":[...],"used_tiebreak":bool,"cost_ratio":float}`

### Voter subprocess contract (`eval`)
Each `--voter CMD` is run via `subprocess.run(shlex.split(CMD), input=task,
capture_output=True, text=True)`. The voter reads the task on **stdin** and
writes its verdict (e.g. `"PASS"`/`"FAIL"`) to **stdout** (stripped). A non-zero
voter exit raises a clear error to stderr.

## Modules

| File | Budget | Responsibility |
|---|---|---|
| `src/loopkit/cli.py` | ~120 lines | `main()` argparse parser; one handler fn per subcommand; `read_input()` helper for `TEXT/@FILE/-`; `run_voter()` helper. |
| `pyproject.toml` | +3 lines | `[project.scripts]` → `loopkit = "loopkit.cli:main"`. |
| `tests/test_cli.py` | ~140 lines | One test per subcommand using `subprocess` against the installed entry point (or `main()` in-process with argv). Cover: JSON shape, exit codes (denied→1, spinning→1), `@file`/`-` input, voter fan-out, stateless vs stateful split, `LOOPKIT_DB` default. |

## Ceiling comments REQUIRED (intentional simplifications)

- Above db default: `# ponytail: hardcoded ~/.loopkit/state.db, no config file — upgrade: add --config if multi-env`
- Above voter execution: `# ponytail: voters run sequentially, not concurrent — upgrade: ThreadPoolExecutor if N>5 and latency-bound`
- Above exit-code mapping: `# ponytail: exit 1 = denied/spinning, no granular codes — upgrade: exit-code map if bash needs reasons`
- Above stdout print: `# ponytail: JSON-only output, no pretty/table — upgrade: --format flag if humans read it`

## Success Criteria

### MUST (blocks ship)
- [ ] `loopkit --help` lists all six subcommands.
- [ ] `loopkit record-outcome --type model --id sonnet --success` then
      `loopkit health` round-trips (state persists in SQLite). — `bash -c '…; loopkit health | jq .'`
- [ ] `loopkit should-dispatch --priority 99 --host deadhost` exits 1 and prints
      `{"allowed": false, ...}` after the breaker is opened by repeated failures.
- [ ] `loopkit check-spin --from <(printf 'a\na\na\na\n')` exits 1 (spinning) and
      prints `mean_ncd` < epsilon. (n outputs yield n−1 NCD values; `window=3`
      therefore needs **4** outputs to evaluate — 3 lines correctly returns
      not-spinning with `mean_ncd: null`.)
- [ ] `loopkit eval --voter 'echo PASS' --voter 'echo PASS' --voter 'echo FAIL'
      --input x` exits 0, verdict `PASS`, `used_tiebreak: false` (binary strict
      majority, no escalation — the v0.2.0 ensemble behavior).
- [ ] `pip install -e .` exposes `loopkit` on PATH.
- [ ] `ruff check src tests` clean; `pytest -q` all green.

### SHOULD (ship with known gaps)
- [ ] `--db` / `LOOPKIT_DB` override works.
- [ ] `@file` and `-` input forms work for `--input`/`--from`.
- [ ] Non-zero voter exit surfaces a clear stderr error.

### NICE (defer)
- [ ] A `loopkit decay` subcommand to apply daily belief decay (not required;
      Orchestrator.decay_all exists; expose only if asked).

## DO NOT BUILD
daemon · HTTP/socket sidecar · click/typer/rich · config/pydantic layer ·
plugin registry · metrics/telemetry · `--format table` · retry/backoff around
voters · a `CliApp`/`Command` base-class abstraction.

## Verification commands
```
cd /tmp/loopkit-audit && .venv/bin/pip install -e . -q
.venv/bin/loopkit --help
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```
