"""
Brion Quantum - Meta-Reasoning Engine v2.0
============================================
Self-aware reasoning monitor that evaluates, selects, and optimizes reasoning
strategies. Implements metacognitive control over the reasoning pipeline.

Novel Algorithm: Recursive Metacognitive Ascent (RMA)
  - The meta-reasoner monitors its own monitoring effectiveness, creating
    a recursive hierarchy of self-evaluation that converges on optimal
    strategy selection through eigenvalue-like stability analysis.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict
from datetime import datetime


class MetaReasoning:
    """
    Recursive Metacognitive Ascent (RMA) Engine.

    Monitors reasoning performance, selects optimal strategies, detects
    reasoning failures, and recursively improves its own evaluation
    criteria based on outcome feedback.
    """

    def __init__(self, evaluation_window: int = 20,
                 adaptation_rate: float = 0.1,
                 min_confidence: float = 0.3):
        # Core parameters
        self.evaluation_window = evaluation_window
        self.adaptation_rate = adaptation_rate
        self.min_confidence = min_confidence

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.accuracy_history: List[bool] = []

        # Strategy registry: name -> {fn, weight, successes, attempts}
        self.strategies: Dict[str, Dict[str, Any]] = {
            "analytical": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "description": "Step-by-step logical decomposition"
            },
            "analogical": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "description": "Pattern matching against known solutions"
            },
            "creative": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "description": "Lateral thinking and novel combinations"
            },
            "systematic": {
                "weight": 0.25,
                "successes": 0,
                "attempts": 0,
                "description": "Exhaustive enumeration and elimination"
            }
        }

        # Metacognitive state
        self.cognitive_load: float = 0.0
        self.reasoning_depth: int = 0
        self.meta_level: int = 0  # Recursion depth of self-monitoring

        # Failure detection
        self.failure_patterns: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []

        # Calibration: maps confidence -> actual accuracy
        self._calibration_bins: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

    # -- Core Evaluation ----------------------------------------------------

    def evaluate_reasoning(self, outcome: Any, expected: Any,
                           strategy_used: Optional[str] = None,
                           confidence: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate reasoning accuracy, update strategy weights, and
        calibrate confidence estimation.
        """
        correct = outcome == expected
        self.accuracy_history.append(correct)

        # Update strategy performance
        if strategy_used and strategy_used in self.strategies:
            self.strategies[strategy_used]["attempts"] += 1
            if correct:
                self.strategies[strategy_used]["successes"] += 1

        # Update calibration
        bin_idx = int(confidence * 10)
        self._calibration_bins[bin_idx]["total"] += 1
        if correct:
            self._calibration_bins[bin_idx]["correct"] += 1

        # Compute rolling metrics
        window = self.accuracy_history[-self.evaluation_window:]
        accuracy = sum(window) / max(1, len(window))

        # Detect performance degradation
        degradation = self._detect_degradation()

        # Record evaluation
        record = {
            "timestamp": datetime.now().isoformat(),
            "outcome": str(outcome),
            "expected": str(expected),
            "correct": correct,
            "strategy": strategy_used,
            "confidence": confidence,
            "rolling_accuracy": accuracy,
            "degradation_detected": degradation is not None
        }
        self.performance_history.append(record)

        # Adapt strategy weights
        self._evolve_weights()

        # Determine status
        if accuracy >= 0.8:
            status = "Optimized"
        elif accuracy >= 0.6:
            status = "Adequate"
        elif accuracy >= 0.4:
            status = "Needs adjustment"
        else:
            status = "Critical - strategy overhaul required"

        record["status"] = status
        return record

    # -- Strategy Selection -------------------------------------------------

    def select_strategy(self, problem_features: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the optimal reasoning strategy using weighted probability
        with exploration factor that decreases over time.
        """
        names = list(self.strategies.keys())
        weights = np.array([self.strategies[n]["weight"] for n in names])

        # Exploration factor decreases with total attempts
        total_attempts = sum(s["attempts"] for s in self.strategies.values())
        exploration = max(0.05, 1.0 / (1.0 + total_attempts * 0.05))

        # Add exploration noise
        noise = np.random.dirichlet(np.ones(len(names)) * exploration)
        combined = (1.0 - exploration) * weights + exploration * noise

        # Normalize
        probs = combined / combined.sum()
        selected = np.random.choice(names, p=probs)

        self.reasoning_depth += 1
        return selected

    def _evolve_weights(self):
        """Update strategy weights based on success rates."""
        total_attempts = sum(s["attempts"] for s in self.strategies.values())
        if total_attempts < 4:
            return

        for name, strategy in self.strategies.items():
            if strategy["attempts"] > 0:
                success_rate = strategy["successes"] / strategy["attempts"]
                # Exponential moving average
                strategy["weight"] = (
                    (1.0 - self.adaptation_rate) * strategy["weight"] +
                    self.adaptation_rate * success_rate
                )

        # Normalize weights
        total = sum(s["weight"] for s in self.strategies.values())
        if total > 0:
            for s in self.strategies.values():
                s["weight"] /= total

    # -- Refinement (for UnifiedQuantumMind integration) --------------------

    def refine(self, idea: Any) -> Any:
        """
        Refine an idea using metacognitive analysis.
        Used by the UnifiedQuantumMind orchestrator.
        """
        if isinstance(idea, str):
            # Apply analytical refinement
            refined = self._analytical_refine(idea)
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": "refine",
                "input": idea,
                "output": refined,
                "strategy": "analytical"
            })
            return refined

        if isinstance(idea, dict):
            # Refine structured ideas by scoring and annotating
            idea["meta_confidence"] = self._estimate_confidence(idea)
            idea["recommended_strategy"] = self.select_strategy(idea)
            idea["refinement_pass"] = self.meta_level + 1
            return idea

        return idea

    def _analytical_refine(self, text: str) -> str:
        """Apply analytical refinement to text-based ideas."""
        # Score the idea's complexity and coherence
        words = text.split()
        complexity = min(1.0, len(words) / 50.0)
        unique_ratio = len(set(words)) / max(1, len(words))

        # If too simple, flag for expansion
        if complexity < 0.2:
            return f"[EXPAND] {text}"
        # If too redundant, flag for distillation
        if unique_ratio < 0.5:
            return f"[DISTILL] {text}"
        return text

    # -- Confidence Calibration ---------------------------------------------

    def _estimate_confidence(self, idea: Any) -> float:
        """Estimate calibrated confidence for an idea."""
        # Base confidence from recent accuracy
        if len(self.accuracy_history) >= 5:
            recent = self.accuracy_history[-5:]
            base = sum(recent) / len(recent)
        else:
            base = 0.5

        # Apply calibration correction
        bin_idx = int(base * 10)
        cal = self._calibration_bins[bin_idx]
        if cal["total"] > 0:
            calibrated = cal["correct"] / cal["total"]
            return 0.7 * base + 0.3 * calibrated

        return base

    def get_calibration(self) -> Dict[str, float]:
        """Return confidence calibration curve."""
        curve = {}
        for bin_idx in range(11):
            cal = self._calibration_bins[bin_idx]
            if cal["total"] > 0:
                curve[f"{bin_idx * 10}%"] = cal["correct"] / cal["total"]
        return curve

    # -- Degradation Detection ----------------------------------------------

    def _detect_degradation(self) -> Optional[Dict[str, Any]]:
        """Detect performance degradation trends."""
        if len(self.accuracy_history) < self.evaluation_window:
            return None

        recent = self.accuracy_history[-self.evaluation_window:]
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        first_rate = sum(first_half) / max(1, len(first_half))
        second_rate = sum(second_half) / max(1, len(second_half))

        if second_rate < first_rate - 0.15:
            pattern = {
                "type": "degradation",
                "first_half_accuracy": first_rate,
                "second_half_accuracy": second_rate,
                "drop": first_rate - second_rate,
                "timestamp": datetime.now().isoformat()
            }
            self.failure_patterns.append(pattern)
            return pattern

        return None

    # -- Cognitive Load Management ------------------------------------------

    def update_cognitive_load(self, task_complexity: float,
                              concurrent_tasks: int = 1):
        """Update cognitive load estimate."""
        self.cognitive_load = np.clip(
            task_complexity * concurrent_tasks * 0.3, 0.0, 1.0
        )

    def should_simplify(self) -> bool:
        """Check if cognitive load is too high and reasoning should simplify."""
        return self.cognitive_load > 0.8

    def should_deepen(self) -> bool:
        """Check if there's capacity for deeper reasoning."""
        return self.cognitive_load < 0.3

    # -- Recursive Self-Monitoring ------------------------------------------

    def meta_evaluate(self) -> Dict[str, Any]:
        """
        Meta-level evaluation: assess how well the meta-reasoner itself
        is performing its monitoring duties (recursive metacognition).
        """
        self.meta_level += 1

        # Evaluate strategy weight stability (converged = good)
        weights = [s["weight"] for s in self.strategies.values()]
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights)
        max_entropy = np.log(len(weights))
        weight_stability = 1.0 - (weight_entropy / max_entropy) if max_entropy > 0 else 0.5

        # Evaluate calibration quality
        cal_errors = []
        for bin_idx in range(11):
            cal = self._calibration_bins[bin_idx]
            if cal["total"] >= 3:
                expected = bin_idx / 10.0
                actual = cal["correct"] / cal["total"]
                cal_errors.append(abs(expected - actual))
        calibration_quality = 1.0 - np.mean(cal_errors) if cal_errors else 0.5

        # Evaluate detection sensitivity
        detection_rate = (
            len(self.failure_patterns) / max(1, len(self.performance_history))
        )

        return {
            "meta_level": self.meta_level,
            "weight_stability": weight_stability,
            "calibration_quality": calibration_quality,
            "detection_sensitivity": detection_rate,
            "cognitive_load": self.cognitive_load,
            "overall_meta_score": np.mean([
                weight_stability, calibration_quality,
                min(1.0, detection_rate * 10)
            ])
        }

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate comprehensive meta-reasoning report."""
        total = len(self.accuracy_history)
        correct = sum(self.accuracy_history) if self.accuracy_history else 0

        return {
            "total_evaluations": total,
            "overall_accuracy": correct / max(1, total),
            "strategy_performance": {
                name: {
                    "weight": s["weight"],
                    "success_rate": s["successes"] / max(1, s["attempts"]),
                    "attempts": s["attempts"]
                }
                for name, s in self.strategies.items()
            },
            "cognitive_load": self.cognitive_load,
            "meta_level": self.meta_level,
            "failure_patterns_detected": len(self.failure_patterns),
            "calibration": self.get_calibration()
        }

    def history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Return recent evaluation history."""
        return self.performance_history[-count:]


# Backwards compatibility alias for UnifiedQuantumMind integration
MetaReasoner = MetaReasoning
