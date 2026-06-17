import unittest

from loopkit.degradation import SystemMetrics
from loopkit.orchestrator import Orchestrator


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.ork = Orchestrator()

    def test_dispatch_allowed_when_healthy(self):
        decision = self.ork.should_dispatch(20, "local")
        self.assertTrue(decision.allowed)

    def test_dispatch_blocked_by_circuit_breaker(self):
        # Trip the breaker
        for _ in range(5):
            self.ork.breakers.record("sandbox", False)
        decision = self.ork.should_dispatch(10, "sandbox")
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.circuit_breaker_open)

    def test_dispatch_blocked_by_degradation(self):
        metrics = SystemMetrics(
            budget_remaining_pct=5,
            host_load=20,
            p_system_healthy=0.1,
        )
        decision = self.ork.should_dispatch(
            40, "local", metrics=metrics
        )
        self.assertFalse(decision.allowed)

    def test_select_model_cold_start(self):
        model = self.ork.select_model(
            ["haiku", "sonnet", "opus"],
            fallback="sonnet",
            min_obs=5,
        )
        self.assertEqual(model, "sonnet")

    def test_select_model_learned(self):
        for _ in range(20):
            self.ork.beliefs.update("model", "opus", True, "debug")
            self.ork.beliefs.update("model", "haiku", False, "debug")
        wins = sum(
            1 for _ in range(50)
            if self.ork.select_model(["opus", "haiku"], "debug") == "opus"
        )
        self.assertGreater(wins, 35)

    def test_select_model_forwards_min_obs_warmup(self):
        # One warm model + one under-tried model: with min_obs warmup the
        # under-tried model is explored rather than Thompson-sampled away.
        for _ in range(20):
            self.ork.beliefs.update("model", "opus", True, "debug")
        self.ork.beliefs.update("model", "haiku", False, "debug")  # 1 obs
        # min_obs=5 forces exploration of haiku (least-tried) every time.
        for _ in range(10):
            self.assertEqual(
                self.ork.select_model(["opus", "haiku"], "debug", min_obs=5),
                "haiku",
            )

    def test_record_outcome(self):
        self.ork.record_outcome("skill", "build", True, host="local")
        b = self.ork.beliefs.get("skill", "build")
        self.assertEqual(b.total_obs, 1)

    def test_spin_detection(self):
        base = "Same output every time"
        for _ in range(5):
            result = self.ork.check_spin("job1", base)
        self.assertTrue(result)

    def test_chain_ok(self):
        self.ork.dag.add_chain(["a", "b", "c"])
        for _ in range(20):
            self.ork.beliefs.update("skill", "a", True)
            self.ork.beliefs.update("skill", "b", True)
            self.ork.beliefs.update("skill", "c", True)
        self.assertTrue(self.ork.chain_ok("a", min_reliability=0.5))

    def test_health_report(self):
        self.ork.beliefs.update("model", "sonnet", True)
        self.ork.cusum.register("test", baseline=1.0)
        report = self.ork.health_report()
        self.assertIn("degradation", report)
        self.assertIn("cusum", report)

    def test_save_and_load(self):
        ork1 = Orchestrator(db_path=":memory:")
        ork1.beliefs.update("skill", "test", True)
        ork1.dag.add_chain(["a", "b"])
        ork1.cusum.register("m1", baseline=1.0)
        ork1.save()

        # Load into new orchestrator from same store
        ork2 = Orchestrator.__new__(Orchestrator)
        ork2.store = ork1.store
        ork2.beliefs = ork2.store.load_beliefs()
        b = ork2.beliefs.get("skill", "test")
        self.assertEqual(b.total_obs, 1)


if __name__ == "__main__":
    unittest.main()
