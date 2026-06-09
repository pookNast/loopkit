import unittest
from loopkit.beliefs import BetaBelief, BeliefEngine


class TestBetaBelief(unittest.TestCase):
    def test_initial_uniform(self):
        b = BetaBelief()
        self.assertAlmostEqual(b.mean, 0.5)

    def test_update_success(self):
        b = BetaBelief()
        b.update(True)
        self.assertGreater(b.mean, 0.5)
        self.assertEqual(b.total_obs, 1)

    def test_update_failure(self):
        b = BetaBelief()
        b.update(False)
        self.assertLess(b.mean, 0.5)

    def test_sample_in_range(self):
        b = BetaBelief(alpha=10, beta=5)
        for _ in range(100):
            s = b.sample()
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_credible_interval(self):
        b = BetaBelief(alpha=50, beta=10)
        lo, hi = b.credible_interval(0.95)
        self.assertLess(lo, b.mean)
        self.assertGreater(hi, b.mean)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_decay_floors_at_one(self):
        b = BetaBelief(alpha=1.0, beta=1.0)
        b.decay(0.5)
        self.assertGreaterEqual(b.alpha, 1.0)
        self.assertGreaterEqual(b.beta, 1.0)

    def test_to_dict(self):
        b = BetaBelief(alpha=3, beta=2)
        d = b.to_dict()
        self.assertIn("alpha", d)
        self.assertIn("ci_95", d)


class TestBeliefEngine(unittest.TestCase):
    def test_get_creates_default(self):
        engine = BeliefEngine()
        b = engine.get("model", "sonnet")
        self.assertAlmostEqual(b.mean, 0.5)

    def test_update_and_retrieve(self):
        engine = BeliefEngine()
        engine.update("model", "sonnet", True)
        engine.update("model", "sonnet", True)
        engine.update("model", "sonnet", False)
        b = engine.get("model", "sonnet")
        self.assertEqual(b.total_obs, 3)
        self.assertGreater(b.mean, 0.5)

    def test_select_best_returns_something(self):
        engine = BeliefEngine()
        for _ in range(10):
            engine.update("model", "opus", True)
        for _ in range(10):
            engine.update("model", "haiku", False)
        # Opus should win most of the time
        wins = sum(
            1 for _ in range(100)
            if engine.select_best("model", ["opus", "haiku"]) == "opus"
        )
        self.assertGreater(wins, 70)

    def test_select_best_empty(self):
        engine = BeliefEngine()
        self.assertIsNone(engine.select_best("model"))

    def test_ranked_by_mean(self):
        engine = BeliefEngine()
        engine.update("skill", "build", True)
        engine.update("skill", "deploy", False)
        ranked = engine.ranked("skill", by="mean")
        self.assertEqual(ranked[0][0], "build")

    def test_context_isolation(self):
        engine = BeliefEngine()
        engine.update("model", "sonnet", True, context="debug")
        engine.update("model", "sonnet", False, context="routine")
        b_debug = engine.get("model", "sonnet", "debug")
        b_routine = engine.get("model", "sonnet", "routine")
        self.assertGreater(b_debug.mean, b_routine.mean)


if __name__ == "__main__":
    unittest.main()
