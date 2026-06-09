import unittest
from loopkit.ensemble import EnsembleEvaluator, condorcet_accuracy


class TestCondorcetAccuracy(unittest.TestCase):
    def test_perfect_voters(self):
        self.assertAlmostEqual(condorcet_accuracy(1.0, 3), 1.0)

    def test_coin_flip_voters(self):
        self.assertAlmostEqual(condorcet_accuracy(0.5, 3), 0.5)

    def test_good_voters_improve(self):
        p1 = 0.7
        p_ensemble = condorcet_accuracy(p1, 3)
        self.assertGreater(p_ensemble, p1)

    def test_even_voters_raises(self):
        with self.assertRaises(ValueError):
            condorcet_accuracy(0.7, 4)


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

    def test_disagreement_triggers_tiebreak(self):
        call_count = [0]

        def cheap(task):
            call_count[0] += 1
            return "PASS" if call_count[0] <= 2 else "FAIL"

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=lambda _: "FINAL",
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertTrue(result.used_tiebreak)
        self.assertEqual(result.verdict, "FINAL")

    def test_no_expensive_fn_uses_majority(self):
        call_count = [0]

        def cheap(task):
            call_count[0] += 1
            return "PASS" if call_count[0] <= 2 else "FAIL"

        evaluator = EnsembleEvaluator(
            cheap_fn=cheap,
            expensive_fn=None,
            n_voters=3,
        )
        result = evaluator.evaluate("task")
        self.assertEqual(result.verdict, "PASS")
        self.assertFalse(result.used_tiebreak)

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


if __name__ == "__main__":
    unittest.main()
