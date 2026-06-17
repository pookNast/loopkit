"""Contract tests for the loopkit CLI (MILESTONE-CONTRACT.md).

Drives every subcommand in-process via ``loopkit.cli.main(argv=[...])`` and
asserts the locked JSON shapes / exit codes. A single subprocess smoke test at
the end proves the console-script entry point is wired (skipped if absent).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


def _run(argv, stdin_data=None):
    """Call cli.main(argv) in-process; capture stdout/stderr + SystemExit code.

    Returns (code, out, err). code is None when main() returned without raising.
    """
    from unittest.mock import patch

    out_buf, err_buf = io.StringIO(), io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            if stdin_data is not None:
                # contextlib has no redirect_stdin; swap sys.stdin directly.
                with patch("sys.stdin", io.StringIO(stdin_data)):
                    from loopkit.cli import main
                    main(argv)
            else:
                from loopkit.cli import main
                main(argv)
    except SystemExit as exc:
        code = exc.code
    return code, out_buf.getvalue(), err_buf.getvalue()


class _TmpDb(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="loopkit_cli_")
        self.db = os.path.join(self._dir, "state.db")

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)


class TestRecordAndHealth(_TmpDb):
    def test_record_outcome_then_health_roundtrip(self):
        code, out, _ = _run([
            "record-outcome", "--db", self.db,
            "--type", "model", "--id", "sonnet", "--success",
        ])
        self.assertEqual(code, 0, out)
        rec = json.loads(out)  # exactly one JSON line
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["entity_type"], "model")
        self.assertEqual(rec["entity_id"], "sonnet")
        self.assertIs(rec["success"], True)

        code, out, _ = _run(["health", "--db", self.db])
        self.assertEqual(code, 0, out)
        report = json.loads(out)
        models = report.get("top_models", [])
        self.assertTrue(any(m["id"] == "sonnet" and m["obs"] >= 1 for m in models),
                        f"sonnet not in top_models: {models}")


class TestSelectModel(_TmpDb):
    def test_returns_one_of_candidates_with_enough_obs(self):
        # Seed both candidates past min_obs so Thompson sampling is real.
        for _ in range(8):
            _run(["record-outcome", "--db", self.db, "--type", "model",
                  "--id", "sonnet", "--success"])
            _run(["record-outcome", "--db", self.db, "--type", "model",
                  "--id", "haiku", "--fail"])
        code, out, _ = _run([
            "select-model", "--db", self.db,
            "--candidates", "sonnet,haiku", "--job-type", "general",
            "--min-obs", "5",
        ])
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        self.assertIn(res["model"], {"sonnet", "haiku"})

    def test_empty_candidates_is_clean_error_code2(self):
        code, out, err = _run([
            "select-model", "--db", self.db,
            "--candidates", "", "--job-type", "general",
        ])
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertIn("no candidates", err)


class TestShouldDispatch(_TmpDb):
    def test_healthy_host_allowed(self):
        code, out, _ = _run([
            "should-dispatch", "--db", self.db,
            "--priority", "10", "--host", "box1",
        ])
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        self.assertIs(res["allowed"], True)
        self.assertIn("circuit_breaker_open", res)
        self.assertIs(res["circuit_breaker_open"], False)

    def test_breaker_opens_after_repeated_failures(self):
        host = "deadhost"
        # Default breaker threshold is 3 consecutive failures -> OPEN.
        for _ in range(4):
            _run(["record-outcome", "--db", self.db, "--type", "host",
                  "--id", host, "--fail", "--host", host])
        code, out, _ = _run([
            "should-dispatch", "--db", self.db,
            "--priority", "99", "--host", host,
        ])
        self.assertEqual(code, 1, out)
        res = json.loads(out)
        self.assertIs(res["allowed"], False)
        self.assertIs(res["circuit_breaker_open"], True)


class TestCheckSpin(_TmpDb):
    def test_identical_outputs_spinning_from_file(self):
        spinfile = os.path.join(self._dir, "lines.txt")
        with open(spinfile, "w") as f:
            f.write("a\na\na\na\n")  # 4 lines -> 3 NCDs at window=3
        code, out, _ = _run([
            "check-spin", "--db", self.db, "--epsilon", "0.15",
            "--window", "3", "--from", spinfile,
        ])
        self.assertEqual(code, 1, out)
        res = json.loads(out)
        self.assertIs(res["spinning"], True)
        self.assertIsNotNone(res["mean_ncd"])
        self.assertLess(res["mean_ncd"], 0.15)

    def test_diverse_outputs_not_spinning_stdin(self):
        code, out, _ = _run(
            ["check-spin", "--db", self.db, "--epsilon", "0.15", "--window", "3"],
            stdin_data="alpha\nbeta\ngamma\ndelta\n",
        )
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        self.assertIs(res["spinning"], False)


class TestEval(_TmpDb):
    def test_majority_no_tiebreak(self):
        code, out, _ = _run([
            "eval", "--db", self.db,
            "--voter", "echo PASS", "--voter", "echo PASS", "--voter", "echo FAIL",
            "--input", "x",
        ])
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        self.assertEqual(res["verdict"], "PASS")
        self.assertIs(res["used_tiebreak"], False)
        self.assertAlmostEqual(res["agreement_ratio"], 0.667, places=2)

    def test_three_way_split_uses_tiebreak(self):
        code, out, _ = _run([
            "eval", "--db", self.db,
            "--voter", "echo A", "--voter", "echo B", "--voter", "echo C",
            "--tiebreak", "echo FINAL",
            "--input", "task-data",
        ])
        self.assertEqual(code, 0, out)
        res = json.loads(out)
        self.assertEqual(res["verdict"], "FINAL")
        self.assertIs(res["used_tiebreak"], True)

    def test_input_at_file(self):
        infile = os.path.join(self._dir, "task.txt")
        with open(infile, "w") as f:
            f.write("file-task")
        code, out, _ = _run([
            "eval", "--db", self.db,
            "--voter", "echo PASS", "--voter", "echo PASS", "--voter", "echo PASS",
            "--input", f"@{infile}",
        ])
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["verdict"], "PASS")

    def test_input_stdin(self):
        code, out, _ = _run(
            ["eval", "--db", self.db,
             "--voter", "echo PASS", "--voter", "echo PASS", "--voter", "echo PASS",
             "--input", "-"],
            stdin_data="stdin-task\n",
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["verdict"], "PASS")


class TestVoterError(_TmpDb):
    def test_nonzero_voter_exit_is_stderr_error_code2(self):
        code, out, err = _run([
            "eval", "--db", self.db,
            "--voter", "sh -c 'exit 3'",
            "--voter", "echo PASS", "--voter", "echo PASS",
            "--input", "x",
        ])
        self.assertEqual(code, 2)
        self.assertTrue(err.strip(), "expected non-empty stderr on voter failure")


class TestLoopkitDbEnv(_TmpDb):
    def test_env_override_used_when_db_omitted(self):
        env_db = os.path.join(self._dir, "env.db")
        saved = os.environ.get("LOOPKIT_DB")
        os.environ["LOOPKIT_DB"] = env_db
        try:
            code, out, _ = _run([
                "record-outcome", "--type", "model", "--id", "opus", "--success",
            ])
            self.assertEqual(code, 0, out)
        finally:
            if saved is None:
                os.environ.pop("LOOPKIT_DB", None)
            else:
                os.environ["LOOPKIT_DB"] = saved
        # State must have landed in the env-specified file.
        self.assertTrue(os.path.exists(env_db),
                        "LOOPKIT_DB override did not create the file")


class TestJsonContract(_TmpDb):
    """Every command prints EXACTLY one JSON line to stdout."""

    def test_single_json_line_stdout(self):
        cmds = [
            ["record-outcome", "--db", self.db, "--type", "model",
             "--id", "x", "--success"],
            ["health", "--db", self.db],
            ["should-dispatch", "--db", self.db, "--priority", "1", "--host", "h"],
        ]
        for argv in cmds:
            code, out, err = _run(argv)
            self.assertEqual(code, 0, f"{argv}: {out}\n{err}")
            out_stripped = out.strip()
            self.assertNotEqual(out_stripped, "")
            obj = json.loads(out_stripped)  # raises if not valid JSON
            # No trailing non-JSON: re-serializing then comparing rules it out.
            self.assertEqual(json.dumps(obj).strip(), out_stripped)


class TestHelpSmoke(unittest.TestCase):
    """Subprocess smoke test for the console-script entry point."""

    def test_help_lists_subcommands(self):
        binpath = os.path.join(os.path.dirname(sys.executable), "loopkit")
        if not os.path.exists(binpath):
            self.skipTest("loopkit console-script not installed yet")
        proc = subprocess.run(
            [binpath, "--help"], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for sub in ("record-outcome", "select-model", "should-dispatch",
                    "health", "check-spin", "eval"):
            self.assertIn(sub, proc.stdout, f"missing {sub} in --help")


if __name__ == "__main__":
    unittest.main()
