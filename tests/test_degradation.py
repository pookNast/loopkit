import unittest

from loopkit.degradation import (
    DegradationLadder,
    DegradationLevel,
    SystemMetrics,
)


class TestDegradationLadder(unittest.TestCase):
    def test_normal_when_healthy(self):
        ladder = DegradationLadder()
        metrics = SystemMetrics()  # all defaults = healthy
        level = ladder.compute(metrics)
        self.assertEqual(level, DegradationLevel.NORMAL)

    def test_constrained_on_failure_spike(self):
        ladder = DegradationLadder()
        metrics = SystemMetrics(failed_24h=15)
        level = ladder.compute(metrics)
        self.assertEqual(level, DegradationLevel.CONSTRAINED)

    def test_stressed_on_budget_low(self):
        ladder = DegradationLadder()
        metrics = SystemMetrics(budget_remaining_pct=10)
        level = ladder.compute(metrics)
        self.assertGreaterEqual(level, DegradationLevel.STRESSED)

    def test_emergency_on_everything_bad(self):
        ladder = DegradationLadder()
        metrics = SystemMetrics(
            budget_remaining_pct=5,
            host_load=20,
            failed_24h=20,
            running=5,
            max_concurrent=6,
            p_system_healthy=0.1,
        )
        level = ladder.compute(metrics)
        self.assertEqual(level, DegradationLevel.EMERGENCY)

    def test_should_dispatch_filters(self):
        ladder = DegradationLadder()
        # At NORMAL, everything dispatches
        self.assertTrue(ladder.should_dispatch(40))
        # Force to CONSTRAINED
        ladder.compute(SystemMetrics(failed_24h=15))
        self.assertTrue(ladder.should_dispatch(20))
        self.assertFalse(ladder.should_dispatch(40))

    def test_effective_model_override(self):
        ladder = DegradationLadder()
        ladder.compute(SystemMetrics(budget_remaining_pct=5))
        self.assertEqual(ladder.effective_model("opus"), "haiku")

    def test_to_dict(self):
        ladder = DegradationLadder()
        d = ladder.to_dict()
        self.assertIn("level", d)
        self.assertIn("name", d)


if __name__ == "__main__":
    unittest.main()
