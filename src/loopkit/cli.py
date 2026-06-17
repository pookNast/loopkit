"""loopkit CLI -- short-lived process per call, one JSON line to stdout."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable

from loopkit.ensemble import EnsembleEvaluator
from loopkit.ncd import SpinDetector
from loopkit.orchestrator import Orchestrator


def read_input(spec: str | None, default: str | None = None) -> str:
    """Resolve a TEXT/@PATH/- input convention to its string content."""
    if spec is None:
        return default or ""
    if spec == "-":
        return sys.stdin.read()
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            return f.read()
    return spec


def run_voter(cmd: str, task: str) -> str:
    """Run one voter subprocess; return stripped stdout, raise on non-zero exit."""
    # ponytail: voters run sequentially, not concurrent — upgrade: ThreadPoolExecutor if N>5 and latency-bound
    proc = subprocess.run(
        shlex.split(cmd), input=task, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"voter exited {proc.returncode}: {cmd!r} stderr={proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _resolve_db(args: argparse.Namespace) -> str:
    db = getattr(args, "db", None) or os.environ.get("LOOPKIT_DB")
    if not db:
        # ponytail: hardcoded ~/.loopkit/state.db, no config file — upgrade: add --config if multi-env
        db = os.path.expanduser("~/.loopkit/state.db")
    directory = os.path.dirname(db)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return db


def _ork(args: argparse.Namespace) -> Orchestrator:
    return Orchestrator(db_path=_resolve_db(args))


def _emit(result: dict) -> None:
    # ponytail: JSON-only output, no pretty/table — upgrade: --format flag if humans read it
    print(json.dumps(result))


def cmd_record_outcome(args: argparse.Namespace) -> int:
    ork = _ork(args)
    success = args.verdict == "success"
    ork.record_outcome(
        entity_type=args.type,
        entity_id=args.id,
        success=success,
        context=args.context,
        host=args.host,
    )
    ork.save()
    _emit(
        {
            "status": "ok",
            "entity_type": args.type,
            "entity_id": args.id,
            "success": success,
        }
    )
    return 0


def cmd_select_model(args: argparse.Namespace) -> int:
    ork = _ork(args)
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    if not candidates:
        raise ValueError("no candidates provided (--candidates is empty)")
    model = ork.select_model(
        candidates,
        job_type=args.job_type,
        min_obs=args.min_obs,
        fallback=args.fallback,
    )
    _emit({"model": model})
    return 0


def cmd_should_dispatch(args: argparse.Namespace) -> int:
    ork = _ork(args)
    decision = ork.should_dispatch(job_priority=args.priority, host=args.host)
    _emit(
        {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "degradation_level": decision.degradation_level.name,
            "model_override": decision.model_override,
            "circuit_breaker_open": decision.circuit_breaker_open,
        }
    )
    # ponytail: exit 1 = denied/spinning, no granular codes — upgrade: exit-code map if bash needs reasons
    return 0 if decision.allowed else 1


def cmd_health(args: argparse.Namespace) -> int:
    ork = _ork(args)
    _emit(ork.health_report())
    return 0


def cmd_check_spin(args: argparse.Namespace) -> int:
    if args.from_file:
        with open(args.from_file) as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()
    detector = SpinDetector(epsilon=args.epsilon, window=args.window)
    spinning = False
    for line in raw.splitlines():
        if line == "":
            continue
        if detector.feed(line):
            spinning = True
    mean_ncd = detector.mean_ncd
    state = detector.to_dict()
    _emit(
        {
            "spinning": spinning,
            "mean_ncd": round(mean_ncd, 4) if mean_ncd is not None else None,
            "spin_count": state["spin_count"],
            "outputs_seen": state["outputs_seen"],
        }
    )
    return 1 if spinning else 0


def cmd_eval(args: argparse.Namespace) -> int:
    task = read_input(args.input, default=None)
    voter_cmds = list(args.voter)
    voter_fns: list[Callable[[str], str]] = [
        (lambda _t, c=cmd: run_voter(c, _t)) for cmd in voter_cmds
    ]
    tie_cmd = args.tiebreak
    tie_fn = (lambda _t: run_voter(tie_cmd, _t)) if tie_cmd else None
    evaluator = EnsembleEvaluator(
        voters=voter_fns,
        expensive_fn=tie_fn,
    )
    result = evaluator.evaluate(task)
    _emit(
        {
            "verdict": result.verdict,
            "agreement_ratio": result.agreement_ratio,
            "votes": result.votes,
            "used_tiebreak": result.used_tiebreak,
            "cost_ratio": result.cost_ratio,
        }
    )
    return 0


def _add_db(p: argparse.ArgumentParser, *, is_parent: bool = False) -> None:
    if is_parent:
        p.add_argument("--db", default=None, help="state DB path")
    else:
        # SUPPRESS: absent flag keeps parent's --db value instead of clobbering it
        p.add_argument("--db", default=argparse.SUPPRESS, help="state DB path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopkit")
    _add_db(parser, is_parent=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record-outcome")
    _add_db(p)
    p.add_argument("--type", required=True)
    p.add_argument("--id", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--success", dest="verdict", action="store_const", const="success")
    group.add_argument("--fail", dest="verdict", action="store_const", const="fail")
    p.add_argument("--context", default="global")
    p.add_argument("--host", default=None)
    p.set_defaults(func=cmd_record_outcome)

    p = sub.add_parser("select-model")
    _add_db(p)
    p.add_argument("--candidates", required=True)
    p.add_argument("--job-type", required=True)
    p.add_argument("--min-obs", type=int, default=5)
    p.add_argument("--fallback", default=None)
    p.set_defaults(func=cmd_select_model)

    p = sub.add_parser("should-dispatch")
    _add_db(p)
    p.add_argument("--priority", type=int, required=True)
    p.add_argument("--host", required=True)
    p.set_defaults(func=cmd_should_dispatch)

    p = sub.add_parser("health")
    _add_db(p)
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("check-spin")
    _add_db(p)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--from", dest="from_file", default=None)
    p.set_defaults(func=cmd_check_spin)

    p = sub.add_parser("eval")
    _add_db(p)
    p.add_argument("--voter", action="append", required=True)
    p.add_argument("--tiebreak", default=None)
    p.add_argument("--input", default=None)
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
    except Exception as exc:
        print(f"loopkit: {exc}", file=sys.stderr)
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
