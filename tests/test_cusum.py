import unittest
from loopkit.cusum import CUSUM, CUSUMBank


class TestCUSUM(unittest.TestCase):
    def test_no_alert_at_baseline(self):
        c = CUSUM(baseline=0.85, _calibrating=False)
        for _ in range(100):
            self.assertFalse(c.update(0.85))

    def test_alert_on_sustained_drift(self):
        c = CUSUM(baseline=0.85, allowance_k=0.05, threshold_h=4.0, _calibrating=False)
        alerted = False
        for _ in range(200):
            if c.update(0.70):  # sustained drop
                alerted = True
                break
        self.assertTrue(alerted)

    def test_no_alert_on_noise(self):
        c = CUSUM(baseline=0.85, allowance_k=0.05, threshold_h=4.0, _calibrating=False)
        import random
        random.seed(42)
        alerts = sum(c.update(0.85 + random.gauss(0, 0.02)) for _ in range(100))
        self.assertLess(alerts, 3)

    def test_distance_to_threshold(self):
        c = CUSUM(baseline=0.85, threshold_h=4.0, _calibrating=False)
        self.assertAlmostEqual(c.distance_to_threshold, 1.0)
        c.update(0.5)
        self.assertLess(c.distance_to_threshold, 1.0)

    def test_auto_calibration(self):
        c = CUSUM(baseline=0.0, _calibrating=True, _cal_target=5)
        for _ in range(5):
            c.update(0.80)
        self.assertAlmostEqual(c.baseline, 0.80)
        self.assertFalse(c._calibrating)

    def test_reset(self):
        c = CUSUM(baseline=0.85, _calibrating=False)
        c.update(0.5)
        c.reset()
        self.assertEqual(c.upper, 0.0)
        self.assertEqual(c.lower, 0.0)


class TestCUSUMBank(unittest.TestCase):
    def test_register_and_update(self):
        bank = CUSUMBank()
        bank.register("success_rate", baseline=0.85)
        result = bank.update("success_rate", 0.85)
        self.assertFalse(result)

    def test_update_many(self):
        bank = CUSUMBank()
        bank.register("cost", baseline=1.0, auto_calibrate=False)
        bank.register("latency", baseline=2.0, auto_calibrate=False)
        results = bank.update_many({"cost": 1.0, "latency": 2.0})
        self.assertFalse(results["cost"])
        self.assertFalse(results["latency"])

    def test_unknown_metric_raises(self):
        bank = CUSUMBank()
        with self.assertRaises(KeyError):
            bank.update("nonexistent", 1.0)

    def test_to_dict(self):
        bank = CUSUMBank()
        bank.register("test", baseline=1.0)
        d = bank.to_dict()
        self.assertIn("test", d)


if __name__ == "__main__":
    unittest.main()
