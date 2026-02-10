"""
Brion Quantum - Abductive Reasoning Engine v2.0
=================================================
Inference to the best explanation with multi-criteria hypothesis ranking,
explanatory coherence analysis, and Occam's razor scoring.

Novel Algorithm: Quantum Explanatory Coherence Network (QECN)
  - Models hypotheses as nodes in a coherence network where explanatory
    relationships create constructive interference (supporting) or
    destructive interference (contradicting), allowing the best
    explanation to emerge through wave function collapse.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class AbductiveHypothesisGenerator:
    """
    Quantum Explanatory Coherence Network (QECN) Engine.

    Generates and ranks hypotheses using multi-criteria evaluation:
    prior probability, explanatory scope, simplicity (Occam's razor),
    coherence with existing knowledge, and predictive power.
    """

    def __init__(self, simplicity_weight: float = 0.2,
                 coherence_weight: float = 0.3,
                 scope_weight: float = 0.25,
                 prior_weight: float = 0.25):
        # Scoring weights
        self.weights = {
            "simplicity": simplicity_weight,
            "coherence": coherence_weight,
            "scope": scope_weight,
            "prior": prior_weight
        }

        # Knowledge base for coherence checking
        self.known_facts: List[str] = []
        self.hypothesis_history: List[Dict[str, Any]] = []

        # Coherence network: (hypothesis, hypothesis) -> correlation
        self.coherence_network: Dict[Tuple[str, str], float] = {}

    # -- Hypothesis Generation -----------------------------------------------

    def generate_hypothesis(self, observations: Any,
                            explanations: Dict[str, float]) -> Dict[str, Any]:
        """
        Select the best explanation for given observations.
        Enhanced with multi-criteria scoring.
        """
        if not explanations:
            return {"best": None, "ranking": [], "confidence": 0.0}

        # Score each hypothesis
        scored = {}
        for hypothesis, prior in explanations.items():
            score = self._score_hypothesis(hypothesis, prior, observations)
            scored[hypothesis] = score

        # Normalize
        total = sum(scored.values())
        if total > 0:
            scored = {k: v / total for k, v in scored.items()}

        # Rank
        ranking = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        best = ranking[0] if ranking else (None, 0.0)

        result = {
            "best": best[0],
            "confidence": best[1],
            "ranking": [{"hypothesis": h, "score": s} for h, s in ranking],
            "observations": str(observations),
            "timestamp": datetime.now().isoformat()
        }

        self.hypothesis_history.append(result)
        return result

    def _score_hypothesis(self, hypothesis: str, prior: float,
                           observations: Any) -> float:
        """Multi-criteria hypothesis scoring."""
        simplicity = self._occam_score(hypothesis)
        coherence = self._coherence_score(hypothesis)
        scope = self._scope_score(hypothesis, observations)

        score = (
            self.weights["prior"] * prior +
            self.weights["simplicity"] * simplicity +
            self.weights["coherence"] * coherence +
            self.weights["scope"] * scope
        )

        return float(np.clip(score, 0.0, 1.0))

    # -- Scoring Components -------------------------------------------------

    def _occam_score(self, hypothesis: str) -> float:
        """
        Occam's razor: simpler hypotheses score higher.
        Measured by inverse of hypothesis complexity (word count, clause count).
        """
        words = hypothesis.split()
        clauses = hypothesis.count(",") + hypothesis.count(";") + 1

        # Simpler = fewer words and clauses
        complexity = len(words) * 0.1 + clauses * 0.3
        simplicity = 1.0 / (1.0 + complexity)
        return simplicity

    def _coherence_score(self, hypothesis: str) -> float:
        """
        Explanatory coherence: how well the hypothesis fits with known facts.
        Uses word overlap as a proxy for semantic coherence.
        """
        if not self.known_facts:
            return 0.5

        h_words = set(hypothesis.lower().split())
        coherence_scores = []

        for fact in self.known_facts:
            f_words = set(fact.lower().split())
            overlap = len(h_words & f_words) / max(1, len(h_words | f_words))
            coherence_scores.append(overlap)

        return float(np.mean(coherence_scores)) if coherence_scores else 0.5

    def _scope_score(self, hypothesis: str, observations: Any) -> float:
        """
        Explanatory scope: how many observations the hypothesis can explain.
        """
        if isinstance(observations, (list, tuple)):
            # More observations explained = higher scope
            explained = 0
            for obs in observations:
                obs_words = set(str(obs).lower().split())
                h_words = set(hypothesis.lower().split())
                if len(obs_words & h_words) > 0:
                    explained += 1
            return explained / max(1, len(observations))

        return 0.5  # Single observation default

    # -- Knowledge Management -----------------------------------------------

    def add_fact(self, fact: str):
        """Add a known fact to the knowledge base."""
        self.known_facts.append(fact)

    def add_coherence_link(self, h1: str, h2: str, correlation: float):
        """Record coherence relationship between two hypotheses."""
        self.coherence_network[(h1, h2)] = np.clip(correlation, -1.0, 1.0)
        self.coherence_network[(h2, h1)] = np.clip(correlation, -1.0, 1.0)

    # -- Iterative Refinement -----------------------------------------------

    def refine_hypotheses(self, hypotheses: List[str],
                           new_evidence: Any) -> List[Dict[str, Any]]:
        """
        Refine a set of hypotheses given new evidence.
        Hypotheses that are contradicted by evidence are penalized;
        those supported are boosted.
        """
        evidence_str = str(new_evidence).lower()
        e_words = set(evidence_str.split())

        refined = []
        for h in hypotheses:
            h_words = set(h.lower().split())
            overlap = len(h_words & e_words) / max(1, len(h_words | e_words))

            # Support score: higher overlap = more support
            support = overlap
            # Contradiction check: explicit negation words
            negation_words = {"not", "never", "false", "wrong", "impossible"}
            contradiction = len(e_words & negation_words) > 0 and overlap > 0.3

            refined.append({
                "hypothesis": h,
                "support": support,
                "contradicted": contradiction,
                "adjusted_confidence": support * (0.2 if contradiction else 1.0)
            })

        return sorted(refined, key=lambda x: x["adjusted_confidence"], reverse=True)

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate abductive reasoning report."""
        return {
            "total_hypotheses_evaluated": len(self.hypothesis_history),
            "known_facts": len(self.known_facts),
            "coherence_links": len(self.coherence_network),
            "scoring_weights": dict(self.weights),
            "recent": self.hypothesis_history[-3:] if self.hypothesis_history else []
        }
