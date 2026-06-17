import unittest

from loopkit.ensemble import EnsembleEvaluator, condorcet_accuracy


class TestCondorcetAccuracy(unittest.TestCase):
    def test_perfect_voters(self):
        self.assertAlmostEqual(condorcet_accuracy(1.0, 3), 1.0)

    def test_coin_flip_voters(self):
        self.assertAlmostEqual(condorcet_accuracy(0.5, 3), 0.5)

    def test_zero_accuracy(self):
        self.assertAlmostEqual(condorcet_accuracy(0.0, 3), 0.0)

    def test_good_voters_improve(self):
        p1 = 0.7
        p_ensemble = condorcet_accuracy(p1, 3)
        self.assertGreater(p_ensemble, p1)

    def test_even_voters_raises(self):
        with self.assertRaises(ValueError):
            condorcet_accuracy(0.7, 4)

    def test_p_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            condorcet_accuracy(-0.1, 3)
        with self.assertRaises(ValueError):
            condorcet_accuracy(1.5, 3)

    def test_n_voters_below_one_raises(self):
        with self.assertRaises(ValueError):
            condorcet_accuracy(0.7, 0)


class TestEnsembleEvaluator(unittest.TestCase):
    def test_unanimous_no_tiebreak(self):
        evaluator = EnsembleEvaluator(
            cheap_fn=lambda _: "PASS",
            expensive_fn=lambda _: "PASS",
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertEqual(result.verdict, "PASS")
        self.assertEqual(result.agreement_ratio, 1.0)
        self.assertFalse(result.used_tiebreak)

    def test_binary_split_has_majority_no_tiebreak(self):
        # 2-1 split: PASS holds a strict majority, so the expensive
        # evaluator must NOT fire under the default policy.
        calls = [0]

        def cheap(task):
            calls[0] += 1
            return "PASS" if calls[0] <= 2 else "FAIL"

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=lambda _: "FINAL",
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertFalse(result.used_tiebreak)
        self.assertEqual(result.verdict, "PASS")
        self.assertAlmostEqual(result.agreement_ratio, 2 / 3, places=2)

    def test_no_majority_triggers_tiebreak(self):
        # Three-way split (no strict majority) -> escalate.
        verdicts = ["A", "B", "C"]

        def cheap(task):
            return verdicts.pop(0)

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=lambda _: "FINAL",
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertTrue(result.used_tiebreak)
        self.assertEqual(result.verdict, "FINAL")

    def test_any_disagreement_policy_escalates_on_split(self):
        # Legacy policy: any non-unanimous vote escalates.
        calls = [0]

        def cheap(task):
            calls[0] += 1
            return "PASS" if calls[0] <= 2 else "FAIL"

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=lambda _: "FINAL",
            n_voters=3,
            escalate_on="any_disagreement",
        )
        result = evaluator.evaluate("task")
        self.assertTrue(result.used_tiebreak)
        self.assertEqual(result.verdict, "FINAL")

    def test_no_expensive_fn_uses_majority(self):
        calls = [0]

        def cheap(task):
            calls[0] += 1
            return "PASS" if calls[0] <= 2 else "FAIL"

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=None,
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertEqual(result.verdict, "PASS")
        self.assertFalse(result.used_tiebreak)

    def test_distinct_voters_called_once_each(self):
        # voters= list gives true independence; each callable invoked exactly once.
        counts = [0, 0, 0]

        def mk(i):
            def fn(task):
                counts[i] += 1
                return "PASS"
            return fn

        evaluator = EnsembleEvaluator(voters=[mk(0), mk(1), mk(2)])
        evaluator.evaluate("task")
        self.assertEqual(counts, [1, 1, 1])

    def test_voters_and_cheap_fn_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            EnsembleEvaluator(
                cheap_fn=lambda _: "PASS",
                voters=[lambda _: "PASS"],
            )

    def test_neither_voters_nor_cheap_fn_raises(self):
        with self.assertRaises(ValueError):
            EnsembleEvaluator()

    def test_cost_ratio(self):
        evaluator = EnsembleEvaluator(
            cheap_fn=lambda _: "PASS",
            n_voters=3,
            cheap_cost=0.2,
            expensive_cost=1.0,
        )
        result = evaluator.evaluate("task")
        self.assertAlmostEqual(result.cost_ratio, 0.6, places=2)

    def test_even_voters_raises(self):
        with self.assertRaises(ValueError):
            EnsembleEvaluator(cheap_fn=lambda _: "PASS", n_voters=4)

    def test_to_dict(self):
        evaluator = EnsembleEvaluator(cheap_fn=lambda _: "PASS")
        evaluator.evaluate("x")
        d = evaluator.to_dict()
        self.assertIn("total_evals", d)
        self.assertEqual(d["n_voters"], 3)
        self.assertEqual(d["escalate_on"], "no_majority")


if __name__ == "__main__":
    unittest.main()
