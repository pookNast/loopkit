import unittest
from loopkit.ncd import ncd, SpinDetector


class TestNCD(unittest.TestCase):
    def test_identical_strings(self):
        d = ncd("hello world", "hello world")
        self.assertLess(d, 0.2)

    def test_different_strings(self):
        d = ncd("hello world", "xyzzy quantum banana")
        self.assertGreater(d, 0.3)

    def test_empty_strings(self):
        self.assertEqual(ncd("", ""), 0.0)

    def test_symmetry(self):
        a, b = "foo bar baz", "quantum computing rocks"
        self.assertAlmostEqual(ncd(a, b), ncd(b, a), places=2)


class TestSpinDetector(unittest.TestCase):
    def test_no_spin_on_novel_outputs(self):
        sd = SpinDetector(epsilon=0.15, window=3)
        outputs = [
            "First attempt: implementing feature A with approach 1",
            "Second attempt: trying approach 2 with different algorithm",
            "Third attempt: complete rewrite using new pattern",
            "Fourth attempt: leveraging external library for solution",
            "Fifth attempt: hybrid approach combining best of all",
        ]
        for out in outputs:
            self.assertFalse(sd.feed(out))

    def test_detects_spin(self):
        sd = SpinDetector(epsilon=0.15, window=3)
        base = "Attempting to fix bug by modifying the handler function"
        spun = False
        for i in range(10):
            # Near-identical outputs with tiny variations
            if sd.feed(base + f" ({i})"):
                spun = True
                break
        self.assertTrue(spun)

    def test_spin_count_increments(self):
        sd = SpinDetector(epsilon=0.15, window=2)
        same = "identical output repeated verbatim"
        for _ in range(5):
            sd.feed(same)
        self.assertGreater(sd.spin_count, 0)

    def test_to_dict(self):
        sd = SpinDetector()
        sd.feed("hello")
        sd.feed("world")
        d = sd.to_dict()
        self.assertIn("last_ncd", d)
        self.assertIn("spin_count", d)


if __name__ == "__main__":
    unittest.main()
