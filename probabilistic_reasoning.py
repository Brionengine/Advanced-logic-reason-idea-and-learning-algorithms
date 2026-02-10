"""
Brion Quantum - Probabilistic Reasoning Engine v2.0
=====================================================
Full Bayesian inference network with evidence accumulation, belief propagation,
and uncertainty quantification. Supports prior updates, likelihood estimation,
and multi-hypothesis reasoning.

Novel Algorithm: Quantum Bayesian Amplitude Estimation (QBAE)
  - Models beliefs as quantum probability amplitudes that interfere
    constructively (confirming evidence) or destructively (conflicting
    evidence), enabling parallel hypothesis evaluation.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class ProbabilisticReasoning:
    """
    Quantum Bayesian Amplitude Estimation (QBAE) Engine.

    Maintains a belief network where hypotheses carry probability amplitudes
    that update via Bayesian rules with quantum-inspired interference effects.
    """

    def __init__(self, prior_strength: float = 1.0,
                 interference_factor: float = 0.1):
        # Core parameters
        self.prior_strength = prior_strength
        self.interference_factor = interference_factor

        # Belief store: hypothesis -> probability amplitude
        self.beliefs: Dict[str, float] = {}

        # Evidence log
        self.evidence_log: List[Dict[str, Any]] = []

        # Conditional probability tables: (hypothesis, evidence) -> P(E|H)
        self.likelihood_table: Dict[Tuple[str, str], float] = {}

        # Inference history
        self.inference_history: List[Dict[str, Any]] = []

    # -- Bayesian Inference -------------------------------------------------

    def bayesian_inference(self, prior: float, likelihood: float,
                           evidence: float) -> float:
        """
        Apply Bayesian update rule to compute posterior probability.
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        if evidence <= 0:
            return prior  # No evidence, return prior unchanged
        posterior = (likelihood * prior) / evidence
        return float(np.clip(posterior, 0.0, 1.0))

    def update_belief(self, hypothesis: str, evidence: str,
                      observation: bool = True) -> float:
        """
        Update belief in a hypothesis given new evidence.
        Uses stored likelihood tables and prior beliefs.
        """
        prior = self.beliefs.get(hypothesis, 0.5)

        # Get likelihood P(E|H) from table or estimate
        if observation:
            likelihood = self.likelihood_table.get((hypothesis, evidence), 0.7)
        else:
            likelihood = 1.0 - self.likelihood_table.get((hypothesis, evidence), 0.7)

        # Compute marginal P(E) over all hypotheses
        marginal = self._compute_marginal(evidence, observation)

        # Bayesian update
        posterior = self.bayesian_inference(prior, likelihood, marginal)

        # Apply quantum interference with related hypotheses
        interference = self._compute_interference(hypothesis, posterior)
        posterior = float(np.clip(posterior + interference, 0.0, 1.0))

        # Store updated belief
        self.beliefs[hypothesis] = posterior

        # Log
        record = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "evidence": evidence,
            "observation": observation,
            "prior": prior,
            "likelihood": likelihood,
            "marginal": marginal,
            "posterior": posterior,
            "interference": interference
        }
        self.evidence_log.append(record)
        self.inference_history.append(record)

        return posterior

    def _compute_marginal(self, evidence: str, observation: bool) -> float:
        """Compute marginal probability P(E) over all hypotheses."""
        if not self.beliefs:
            return 0.5

        total = 0.0
        for hypothesis, prior in self.beliefs.items():
            if observation:
                lik = self.likelihood_table.get((hypothesis, evidence), 0.5)
            else:
                lik = 1.0 - self.likelihood_table.get((hypothesis, evidence), 0.5)
            total += lik * prior

        return max(0.001, total)

    def _compute_interference(self, hypothesis: str,
                               posterior: float) -> float:
        """
        Compute quantum-inspired interference from related hypotheses.
        Hypotheses that are mutually exclusive create destructive interference;
        hypotheses that are correlated create constructive interference.
        """
        if not self.beliefs or self.interference_factor == 0:
            return 0.0

        interference = 0.0
        for other, other_belief in self.beliefs.items():
            if other == hypothesis:
                continue

            # Simple heuristic: hypotheses with similar beliefs interfere
            # constructively, opposing beliefs interfere destructively
            diff = abs(posterior - other_belief)
            if diff < 0.3:
                # Constructive: similar beliefs reinforce
                interference += self.interference_factor * (0.3 - diff)
            else:
                # Destructive: opposing beliefs diminish
                interference -= self.interference_factor * (diff - 0.3)

        return float(np.clip(interference, -0.1, 0.1))

    # -- Multi-Hypothesis Reasoning -----------------------------------------

    def evaluate_hypotheses(self, hypotheses: List[str],
                            evidence_list: List[Tuple[str, bool]]) -> Dict[str, float]:
        """
        Evaluate multiple hypotheses against accumulated evidence.
        Returns posterior probabilities for each hypothesis.
        """
        # Initialize beliefs for new hypotheses
        for h in hypotheses:
            if h not in self.beliefs:
                self.beliefs[h] = 1.0 / len(hypotheses)  # Uniform prior

        # Update each hypothesis with each piece of evidence
        for evidence, observed in evidence_list:
            for h in hypotheses:
                self.update_belief(h, evidence, observed)

        # Normalize beliefs to form proper distribution
        total = sum(self.beliefs[h] for h in hypotheses)
        if total > 0:
            normalized = {h: self.beliefs[h] / total for h in hypotheses}
        else:
            normalized = {h: 1.0 / len(hypotheses) for h in hypotheses}

        return normalized

    def most_likely(self, hypotheses: Optional[List[str]] = None) -> Tuple[str, float]:
        """Return the most likely hypothesis."""
        candidates = {h: p for h, p in self.beliefs.items()
                      if hypotheses is None or h in hypotheses}
        if not candidates:
            return ("unknown", 0.0)
        best = max(candidates.items(), key=lambda x: x[1])
        return best

    # -- Likelihood Management ----------------------------------------------

    def set_likelihood(self, hypothesis: str, evidence: str,
                       probability: float):
        """Set conditional probability P(evidence | hypothesis)."""
        self.likelihood_table[(hypothesis, evidence)] = np.clip(probability, 0.0, 1.0)

    def set_prior(self, hypothesis: str, probability: float):
        """Set prior probability for a hypothesis."""
        self.beliefs[hypothesis] = np.clip(probability, 0.0, 1.0)

    # -- Scoring (for UnifiedQuantumMind integration) -----------------------

    def score(self, idea: Any) -> float:
        """
        Score an idea's probabilistic plausibility.
        Used by the UnifiedQuantumMind orchestrator.
        """
        if isinstance(idea, str):
            # Check if idea matches any high-belief hypothesis
            if idea in self.beliefs:
                return self.beliefs[idea]
            # Score based on similarity to known beliefs
            return self._score_by_similarity(idea)

        if isinstance(idea, dict):
            # Score structured ideas by their component probabilities
            scores = []
            for key, value in idea.items():
                if isinstance(value, (int, float)):
                    scores.append(float(value))
                elif isinstance(value, str) and value in self.beliefs:
                    scores.append(self.beliefs[value])
            return float(np.mean(scores)) if scores else 0.5

        return 0.5  # Default uncertainty

    def _score_by_similarity(self, text: str) -> float:
        """Score text by word overlap with known hypotheses."""
        if not self.beliefs:
            return 0.5

        text_words = set(text.lower().split())
        best_score = 0.5

        for hypothesis, belief in self.beliefs.items():
            h_words = set(hypothesis.lower().split())
            overlap = len(text_words & h_words) / max(1, len(text_words | h_words))
            if overlap > 0.3:
                best_score = max(best_score, belief * overlap)

        return best_score

    # -- Uncertainty Quantification -----------------------------------------

    def entropy(self, hypotheses: Optional[List[str]] = None) -> float:
        """Calculate Shannon entropy of the belief distribution."""
        candidates = {h: p for h, p in self.beliefs.items()
                      if hypotheses is None or h in hypotheses}
        if not candidates:
            return 0.0

        probs = np.array(list(candidates.values()))
        probs = probs / probs.sum()  # Normalize
        # Shannon entropy
        return float(-np.sum(probs * np.log2(probs + 1e-10)))

    def uncertainty(self, hypothesis: str) -> float:
        """Compute uncertainty for a specific hypothesis (variance-like)."""
        p = self.beliefs.get(hypothesis, 0.5)
        # Binary entropy
        return float(-p * np.log2(p + 1e-10) - (1 - p) * np.log2(1 - p + 1e-10))

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate probabilistic reasoning report."""
        return {
            "total_hypotheses": len(self.beliefs),
            "total_evidence_updates": len(self.evidence_log),
            "beliefs": dict(self.beliefs),
            "entropy": self.entropy(),
            "most_likely": self.most_likely(),
            "likelihood_rules": len(self.likelihood_table)
        }


# Backwards compatibility alias for UnifiedQuantumMind integration
ProbabilisticEngine = ProbabilisticReasoning
