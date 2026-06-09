import unittest
from loopkit.beliefs import BeliefEngine
from loopkit.cusum import CUSUMBank
from loopkit.circuit_breaker import CircuitBreakerRegistry
from loopkit.skill_dag import SkillDAG
from loopkit.store import SQLiteStore


class TestSQLiteStore(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteStore(":memory:")

    def test_beliefs_roundtrip(self):
        engine = BeliefEngine()
        engine.update("model", "opus", True)
        engine.update("model", "opus", True)
        engine.update("skill", "build", False)
        self.store.save_beliefs(engine)
        loaded = self.store.load_beliefs()
        self.assertEqual(loaded.get("model", "opus").total_obs, 2)
        self.assertEqual(loaded.get("skill", "build").total_obs, 1)

    def test_cusum_roundtrip(self):
        bank = CUSUMBank()
        bank.register("success_rate", baseline=0.85, auto_calibrate=False)
        bank.update("success_rate", 0.80)
        bank.update("success_rate", 0.75)
        self.store.save_cusum(bank)
        loaded = self.store.load_cusum()
        det = loaded.get("success_rate")
        self.assertEqual(det.observation_count, 2)
        self.assertAlmostEqual(det.baseline, 0.85)

    def test_circuit_breakers_roundtrip(self):
        reg = CircuitBreakerRegistry(default_threshold=2)
        reg.record("sandbox", False)
        reg.record("sandbox", False)
        self.store.save_circuit_breakers(reg)
        loaded = self.store.load_circuit_breakers()
        self.assertIn("sandbox", loaded.open_services())

    def test_skill_graph_roundtrip(self):
        beliefs = BeliefEngine()
        dag = SkillDAG(beliefs)
        dag.add_chain(["a", "b", "c"])
        self.store.save_skill_graph(dag)
        loaded = self.store.load_skill_graph(beliefs)
        self.assertEqual(loaded.walk_chain("a"), ["a", "b", "c"])

    def test_events(self):
        self.store.log_event("test", "started", {"key": "value"})
        events = self.store.recent_events(10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "test")
        self.assertEqual(events[0]["data"]["key"], "value")


if __name__ == "__main__":
    unittest.main()
