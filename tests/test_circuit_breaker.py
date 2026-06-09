import unittest
from loopkit.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, State


class TestCircuitBreaker(unittest.TestCase):
    def test_starts_closed(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, State.CLOSED)
        self.assertTrue(cb.should_attempt())

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, State.OPEN)
        self.assertFalse(cb.should_attempt())

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, State.CLOSED)
        self.assertEqual(cb.failures, 0)

    def test_p_healthy_tracks(self):
        cb = CircuitBreaker()
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        self.assertGreater(cb.p_healthy, 0.5)

    def test_bayesian_cooldown_increases_for_unreliable(self):
        cb = CircuitBreaker(threshold=1, base_cooldown=60)
        for _ in range(10):
            cb.record_failure()
        # After many failures, p_healthy is low, so cooldown should be high
        self.assertGreater(cb._cooldown, 60)


class TestCircuitBreakerRegistry(unittest.TestCase):
    def test_get_creates_default(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get("sandbox")
        self.assertEqual(cb.state, State.CLOSED)

    def test_record_and_open_services(self):
        reg = CircuitBreakerRegistry(default_threshold=2)
        reg.record("sandbox", False)
        reg.record("sandbox", False)
        self.assertIn("sandbox", reg.open_services())

    def test_to_dict(self):
        reg = CircuitBreakerRegistry()
        reg.record("host1", True)
        d = reg.to_dict()
        self.assertIn("host1", d)


if __name__ == "__main__":
    unittest.main()
