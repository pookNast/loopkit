import unittest

from loopkit.beliefs import BeliefEngine
from loopkit.skill_dag import SkillDAG


class TestSkillDAG(unittest.TestCase):
    def setUp(self):
        self.beliefs = BeliefEngine()
        self.dag = SkillDAG(self.beliefs)
        self.dag.add_chain(["build", "test", "deploy", "verify"])

    def test_chain_structure(self):
        self.assertEqual(self.dag.roots(), ["build"])
        self.assertEqual(self.dag.children("build"), ["test"])
        self.assertEqual(self.dag.parents("test"), ["build"])

    def test_walk_chain(self):
        chain = self.dag.walk_chain("build")
        self.assertEqual(chain, ["build", "test", "deploy", "verify"])

    def test_chain_reliability_uniform(self):
        # All skills at Beta(1,1) = uniform => E[theta] = 0.5
        rel = self.dag.chain_reliability("build")
        self.assertAlmostEqual(rel.point_estimate, 0.5**4, places=2)
        self.assertEqual(len(rel.chain), 4)

    def test_chain_reliability_strong(self):
        # Give all skills good track record
        for skill in ["build", "test", "deploy", "verify"]:
            for _ in range(20):
                self.beliefs.update("skill", skill, True)
        rel = self.dag.chain_reliability("build")
        self.assertGreater(rel.point_estimate, 0.5)

    def test_bottleneck_identification(self):
        for _ in range(20):
            self.beliefs.update("skill", "build", True)
            self.beliefs.update("skill", "test", True)
            self.beliefs.update("skill", "deploy", False)  # weak link
            self.beliefs.update("skill", "verify", True)
        rel = self.dag.chain_reliability("build")
        self.assertEqual(rel.bottleneck, "deploy")

    def test_highest_uncertainty(self):
        # Only train some skills
        for _ in range(50):
            self.beliefs.update("skill", "build", True)
        # Others are untrained (high variance)
        skill = self.dag.highest_uncertainty_skill()
        self.assertIn(skill, ["test", "deploy", "verify"])

    def test_to_dict(self):
        d = self.dag.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertEqual(len(d["nodes"]), 4)


if __name__ == "__main__":
    unittest.main()
