"""
Brion Quantum - Epistemic Confidence Engine v2.0
==================================================
Multi-dimensional confidence assessment with calibration tracking,
uncertainty decomposition (aleatoric vs epistemic), and adaptive
thresholding for high-stakes decision making.

Novel Algorithm: Quantum Confidence Superposition (QCS)
  - Models confidence as a superposition of multiple evidence dimensions
    that collapse to a definite value only upon evaluation, capturing
    the inherent uncertainty of pre-measurement knowledge states.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class EpistemicConfidence:
    """
    Quantum Confidence Superposition (QCS) Engine.

    Assesses confidence across multiple dimensions (evidence strength,
    source reliability, internal consistency, temporal relevance) and
    provides calibrated uncertainty estimates.
    """

    def __init__(self, base_threshold: float = 0.7,
                 calibration_window: int = 50,
                 adaptive: bool = True):
        # Core parameters
        self.base_threshold = base_threshold
        self.calibration_window = calibration_window
        self.adaptive = adaptive

        # Dynamic threshold that adjusts based on calibration
        self.current_threshold = base_threshold

        # Confidence dimensions
        self.dimensions = {
            "evidence_strength": 0.25,
            "source_reliability": 0.25,
            "internal_consistency": 0.25,
            "temporal_relevance": 0.25
        }

        # Calibration tracking
        self.calibration_log: List[Dict[str, Any]] = []
        self._confidence_outcomes: List[Tuple[float, bool]] = []

        # Assessment history
        self.assessment_log: List[Dict[str, Any]] = []

    # -- Core Assessment ----------------------------------------------------

    def assess_confidence(self, statement: Any,
                          confidence_level: float) -> str:
        """
        Assess confidence and return statement or warning.
        Backwards compatible with original API.
        """
        calibrated = self._calibrate(confidence_level)

        if calibrated >= self.current_threshold:
            return statement if isinstance(statement, str) else str(statement)
        elif calibrated >= self.current_threshold * 0.5:
            return f"Low confidence ({calibrated:.2f}): {statement}"
        else:
            return "Confidence too low"

    def multi_dimensional_assess(self, statement: Any,
                                  evidence: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Full multi-dimensional confidence assessment using QCS.

        Args:
            statement: The claim or idea to assess
            evidence: Optional dict with dimension scores (0-1):
                - evidence_strength: How strong is the supporting evidence?
                - source_reliability: How reliable are the sources?
                - internal_consistency: Is it self-consistent?
                - temporal_relevance: Is the information current?
        """
        if evidence is None:
            evidence = {}

        # Score each dimension
        dim_scores = {}
        for dim, weight in self.dimensions.items():
            if dim in evidence:
                dim_scores[dim] = float(np.clip(evidence[dim], 0.0, 1.0))
            else:
                dim_scores[dim] = 0.5  # Uncertain by default

        # Compute composite confidence (weighted sum)
        composite = sum(
            dim_scores[dim] * self.dimensions[dim]
            for dim in self.dimensions
        )

        # Compute uncertainty decomposition
        aleatoric = self._aleatoric_uncertainty(dim_scores)
        epistemic = self._epistemic_uncertainty(dim_scores)
        total_uncertainty = aleatoric + epistemic

        # Decision
        calibrated = self._calibrate(composite)
        accepted = calibrated >= self.current_threshold

        result = {
            "statement": str(statement),
            "composite_confidence": composite,
            "calibrated_confidence": calibrated,
            "accepted": accepted,
            "dimension_scores": dim_scores,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total_uncertainty,
            "threshold": self.current_threshold,
            "timestamp": datetime.now().isoformat()
        }

        self.assessment_log.append(result)
        return result

    # -- Evaluation (for UnifiedQuantumMind integration) --------------------

    def evaluate(self, idea: Any) -> float:
        """
        Evaluate confidence in an idea.
        Used by the UnifiedQuantumMind orchestrator.
        Returns confidence score (0-1).
        """
        if isinstance(idea, str):
            # Heuristic confidence based on specificity and length
            words = idea.split()
            specificity = min(1.0, len(set(words)) / max(1, len(words)))
            complexity = min(1.0, len(words) / 30.0)
            base_confidence = 0.5 + 0.3 * specificity + 0.2 * complexity
            return self._calibrate(base_confidence)

        if isinstance(idea, dict):
            # Use provided evidence dimensions if available
            evidence = {k: v for k, v in idea.items()
                        if k in self.dimensions and isinstance(v, (int, float))}
            if evidence:
                result = self.multi_dimensional_assess(idea, evidence)
                return result["calibrated_confidence"]

            # Fallback: average of numeric values
            numeric_vals = [v for v in idea.values()
                           if isinstance(v, (int, float))]
            if numeric_vals:
                return self._calibrate(float(np.mean(numeric_vals)))

        return 0.5  # Default uncertainty

    # -- Calibration --------------------------------------------------------

    def _calibrate(self, raw_confidence: float) -> float:
        """
        Apply calibration correction to raw confidence.
        Uses historical accuracy data to correct systematic over/under-confidence.
        """
        if len(self._confidence_outcomes) < 10:
            return raw_confidence  # Not enough data to calibrate

        # Find similar historical predictions and their outcomes
        nearby = [
            (conf, outcome) for conf, outcome in self._confidence_outcomes
            if abs(conf - raw_confidence) < 0.15
        ]

        if len(nearby) < 3:
            return raw_confidence

        # Actual success rate for similar confidence levels
        actual_rate = sum(1 for _, o in nearby if o) / len(nearby)

        # Blend raw with calibrated
        return 0.6 * raw_confidence + 0.4 * actual_rate

    def record_outcome(self, confidence: float, was_correct: bool):
        """Record an outcome for calibration improvement."""
        self._confidence_outcomes.append((confidence, was_correct))

        # Keep only recent window
        if len(self._confidence_outcomes) > self.calibration_window * 2:
            self._confidence_outcomes = self._confidence_outcomes[-self.calibration_window:]

        # Adaptive threshold adjustment
        if self.adaptive and len(self._confidence_outcomes) >= 20:
            self._adapt_threshold()

        self.calibration_log.append({
            "confidence": confidence,
            "correct": was_correct,
            "timestamp": datetime.now().isoformat()
        })

    def _adapt_threshold(self):
        """Adaptively adjust the confidence threshold."""
        recent = self._confidence_outcomes[-20:]
        # Find the threshold that best separates correct from incorrect
        best_threshold = self.base_threshold
        best_f1 = 0.0

        for t in np.arange(0.3, 0.95, 0.05):
            tp = sum(1 for c, o in recent if c >= t and o)
            fp = sum(1 for c, o in recent if c >= t and not o)
            fn = sum(1 for c, o in recent if c < t and o)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(0.001, precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # Smooth update
        self.current_threshold = (
            0.8 * self.current_threshold + 0.2 * best_threshold
        )

    # -- Uncertainty Decomposition ------------------------------------------

    def _aleatoric_uncertainty(self, dim_scores: Dict[str, float]) -> float:
        """
        Aleatoric uncertainty: inherent randomness in the data.
        High when dimension scores cluster near 0.5 (maximum entropy).
        """
        entropies = []
        for score in dim_scores.values():
            # Binary entropy
            p = np.clip(score, 0.01, 0.99)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            entropies.append(entropy)
        return float(np.mean(entropies))

    def _epistemic_uncertainty(self, dim_scores: Dict[str, float]) -> float:
        """
        Epistemic uncertainty: lack of knowledge (reducible with more data).
        High when dimensions disagree with each other.
        """
        scores = list(dim_scores.values())
        if len(scores) < 2:
            return 0.5
        return float(np.std(scores))

    # -- Reporting ----------------------------------------------------------

    def get_calibration_curve(self) -> Dict[str, float]:
        """Generate calibration curve showing predicted vs actual accuracy."""
        bins = defaultdict(lambda: {"correct": 0, "total": 0})

        for confidence, correct in self._confidence_outcomes:
            bin_idx = int(confidence * 10)
            bins[bin_idx]["total"] += 1
            if correct:
                bins[bin_idx]["correct"] += 1

        curve = {}
        for bin_idx in sorted(bins.keys()):
            data = bins[bin_idx]
            if data["total"] > 0:
                predicted = (bin_idx + 0.5) / 10.0
                actual = data["correct"] / data["total"]
                curve[f"{predicted:.1f}"] = actual

        return curve

    def report(self) -> Dict[str, Any]:
        """Generate confidence assessment report."""
        total = len(self._confidence_outcomes)
        correct = sum(1 for _, o in self._confidence_outcomes if o)

        return {
            "total_assessments": len(self.assessment_log),
            "calibration_samples": total,
            "overall_accuracy": correct / max(1, total),
            "current_threshold": self.current_threshold,
            "dimension_weights": dict(self.dimensions),
            "calibration_curve": self.get_calibration_curve()
        }


# Backwards compatibility alias for UnifiedQuantumMind integration
ConfidenceEvaluator = EpistemicConfidence
