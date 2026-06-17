import sqlite3
import unittest

from loopkit.beliefs import BeliefEngine
from loopkit.circuit_breaker import CircuitBreakerRegistry
from loopkit.cusum import CUSUMBank
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

    def test_save_all_is_atomic(self):
        # save_all writes every gem in one transaction; verify all four
        # round-trip after a single call.
        engine = BeliefEngine()
        engine.update("model", "opus", True)
        bank = CUSUMBank()
        bank.register("rate", baseline=0.9, auto_calibrate=False)
        bank.update("rate", 0.8)
        reg = CircuitBreakerRegistry(default_threshold=2)
        reg.record("sandbox", False)
        reg.record("sandbox", False)
        dag = SkillDAG(engine)
        dag.add_chain(["a", "b"])
        self.store.save_all(engine=engine, cusum=bank, breakers=reg, dag=dag)
        self.assertEqual(self.store.load_beliefs().get("model", "opus").total_obs, 1)
        self.assertEqual(self.store.load_cusum().get("rate").observation_count, 1)
        self.assertIn("sandbox", self.store.load_circuit_breakers().open_services())
        self.assertEqual(self.store.load_skill_graph(engine).walk_chain("a"), ["a", "b"])

    def test_save_all_rollback_on_error(self):
        # A bad argument type should roll back the whole transaction,
        # leaving previously-saved beliefs intact.
        engine = BeliefEngine()
        engine.update("model", "opus", True)
        self.store.save_beliefs(engine)
        before = self.store.load_beliefs().get("model", "opus").total_obs
        with self.assertRaises((AttributeError, TypeError, sqlite3.Error)):
            # Pass an object without _edges to force an error mid-write.
            self.store.save_all(engine=engine, dag=object())  # type: ignore[arg-type]
        after = self.store.load_beliefs().get("model", "opus").total_obs
        self.assertEqual(before, after)

    def test_context_manager_closes_connection(self):
        with SQLiteStore(":memory:") as store:
            store.log_event("x", "y")
            self.assertIsNotNone(store._conn)
        self.assertIsNone(store._conn)

    def test_thread_safe_concurrent_writes(self):
        # The store serialises DB writes across threads (check_same_thread=False
        # + lock). Each thread saves its own snapshot; only the store is shared.
        import threading
        errors = []

        def worker(n):
            try:
                eng = BeliefEngine()
                for _ in range(20):
                    eng.update("model", f"m{n}", True)
                    self.store.save_beliefs(eng)
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        # Concurrent reader: must never raise and must always observe a
        # self-consistent snapshot (a single model whose obs count is valid).
        def reader():
            for _ in range(80):
                try:
                    loaded = self.store.load_beliefs()
                    models = [
                        eid for (etype, eid, ctx) in loaded._beliefs if etype == "model"
                    ]
                    # Snapshot-replace => at most one model at a time.
                    assert len(models) <= 1, f"reader saw split state: {models}"
                except Exception as e:  # pragma: no cover - failure path
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        loaded = self.store.load_beliefs()
        # Snapshot-replace semantics: each save_beliefs wipes + rewrites the
        # table, so the survivors are the last writer's state -- not an
        # amalgamation. The concurrency guarantee is that no write raised
        # and the table is left internally consistent.
        models = {eid for (etype, eid, ctx) in loaded._beliefs if etype == "model"}
        self.assertEqual(len(models), 1)
        # Surviving model has the expected observation count.
        survivor = loaded.get("model", next(iter(models)))
        self.assertEqual(survivor.total_obs, 20)


if __name__ == "__main__":
    unittest.main()
