"""loopkit -- Probabilistic orchestration primitives for AI agent loops.

Zero dependencies. Python 3.10+ stdlib only.

Seven gems:
    1. BetaBelief / BeliefEngine  -- Bayesian belief tracking + Thompson Sampling
    2. CUSUM                      -- Cumulative sum anomaly detection
    3. NCD                        -- Normalized Compression Distance for spin detection
    4. CircuitBreaker             -- Three-state FSM with Bayesian cooldown
    5. SkillDAG                   -- Skill dependency graph with chain reliability
    6. DegradationLadder          -- Graceful 5-level degradation hierarchy
    7. EnsembleEvaluator          -- Condorcet majority-vote evaluation

Compose them with Orchestrator for a full probabilistic-core / deterministic-shell system.
"""

from loopkit.beliefs import BeliefEngine, BetaBelief
from loopkit.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from loopkit.cusum import CUSUM, CUSUMBank
from loopkit.degradation import DegradationLadder, DegradationLevel
from loopkit.ensemble import EnsembleEvaluator
from loopkit.ncd import SpinDetector, ncd
from loopkit.orchestrator import Orchestrator
from loopkit.skill_dag import SkillDAG
from loopkit.store import SQLiteStore

__version__ = "0.2.0"

__all__ = [
    "BetaBelief",
    "BeliefEngine",
    "CUSUM",
    "CUSUMBank",
    "ncd",
    "SpinDetector",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "SkillDAG",
    "DegradationLadder",
    "DegradationLevel",
    "EnsembleEvaluator",
    "Orchestrator",
    "SQLiteStore",
]
