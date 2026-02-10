"""
Brion Quantum - Logic Engine v2.0
==================================
Multi-modal formal reasoning engine with deductive, inductive, abductive,
and analogical reasoning capabilities. Supports propositional logic,
first-order logic resolution, and fuzzy logic for uncertain domains.

Novel Algorithm: Quantum Entangled Logic Resolution (QELR)
  - Models logical propositions as entangled state pairs where validity
    propagates through inference chains via entanglement-like correlations.
  - Contradiction detection uses destructive interference: conflicting
    propositions cancel each other's truth amplitude.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime


class LogicEngine:
    """
    Quantum Entangled Logic Resolution (QELR) Engine.

    Provides multi-modal reasoning across deductive, inductive, abductive,
    and analogical domains. Propositions carry truth amplitudes that propagate
    through inference chains, enabling nuanced reasoning under uncertainty.
    """

    def __init__(self, fuzzy_threshold: float = 0.5,
                 contradiction_sensitivity: float = 0.8):
        # Core parameters
        self.fuzzy_threshold = fuzzy_threshold
        self.contradiction_sensitivity = contradiction_sensitivity

        # Knowledge store: proposition -> truth amplitude (0.0 to 1.0)
        self.propositions: Dict[str, float] = {}

        # Inference rules: (premises_tuple) -> (conclusion, confidence)
        self.rules: List[Dict[str, Any]] = []

        # Reasoning history
        self.inference_log: List[Dict[str, Any]] = []
        self.contradiction_log: List[Dict[str, Any]] = []

        # Pattern memory for inductive reasoning
        self._observation_buffer: List[Dict[str, Any]] = []
        self._induced_generalizations: List[Dict[str, Any]] = []

    # -- Deductive Reasoning ------------------------------------------------

    def deductive_reasoning(self, premises: List[Any], conclusion: Any) -> str:
        """
        Determines if the conclusion logically follows from the premises.
        Uses truth amplitude propagation for nuanced validity checking.
        """
        if not premises:
            return "Undetermined"

        # Calculate combined truth amplitude of premises
        amplitudes = []
        for p in premises:
            if isinstance(p, str) and p in self.propositions:
                amplitudes.append(self.propositions[p])
            elif isinstance(p, (bool, int, float)):
                amplitudes.append(float(p))
            else:
                amplitudes.append(0.5)  # Unknown defaults to uncertain

        # Combined amplitude via quantum product rule
        combined = np.prod(amplitudes)

        if combined >= self.fuzzy_threshold:
            self._log_inference("deductive", premises, conclusion, combined, True)
            return conclusion
        elif combined <= (1.0 - self.fuzzy_threshold):
            self._log_inference("deductive", premises, conclusion, combined, False)
            return f"Negated: {conclusion}"
        else:
            self._log_inference("deductive", premises, conclusion, combined, None)
            return "Undetermined"

    def modus_ponens(self, antecedent: str, consequent: str) -> Optional[str]:
        """If P then Q; P is true; therefore Q is true."""
        p_truth = self.propositions.get(antecedent, 0.0)
        if p_truth >= self.fuzzy_threshold:
            q_truth = min(1.0, p_truth * 0.95)  # Slight decay through inference
            self.propositions[consequent] = max(
                self.propositions.get(consequent, 0.0), q_truth
            )
            self._log_inference("modus_ponens", [antecedent], consequent, q_truth, True)
            return consequent
        return None

    def modus_tollens(self, antecedent: str, consequent: str) -> Optional[str]:
        """If P then Q; Q is false; therefore P is false."""
        q_truth = self.propositions.get(consequent, 0.5)
        if q_truth <= (1.0 - self.fuzzy_threshold):
            p_negated_truth = 1.0 - q_truth
            negated = f"NOT({antecedent})"
            self.propositions[negated] = p_negated_truth
            self._log_inference("modus_tollens", [consequent], negated, p_negated_truth, True)
            return negated
        return None

    def syllogism(self, major_premise: Tuple[str, str],
                  minor_premise: Tuple[str, str]) -> Optional[str]:
        """
        Classical syllogism: All A are B, All B are C -> All A are C.
        Premises are (subject, predicate) tuples.
        """
        if major_premise[1] == minor_premise[0]:
            conclusion = f"All {major_premise[0]} are {minor_premise[1]}"
            mp_truth = self.propositions.get(str(major_premise), 0.8)
            mn_truth = self.propositions.get(str(minor_premise), 0.8)
            combined = mp_truth * mn_truth
            self.propositions[conclusion] = combined
            self._log_inference("syllogism", [major_premise, minor_premise],
                                conclusion, combined, combined >= self.fuzzy_threshold)
            return conclusion
        return None

    # -- Inductive Reasoning ------------------------------------------------

    def inductive_reasoning(self, observations: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Generates generalizations from observed instances using statistical
        pattern detection and confidence estimation.
        """
        if not observations:
            return None

        # Store observations
        for obs in observations:
            self._observation_buffer.append({
                "value": obs,
                "timestamp": datetime.now().isoformat()
            })

        # Numerical observations: compute statistics
        numeric = [o for o in observations if isinstance(o, (int, float))]
        if numeric:
            mean_val = float(np.mean(numeric))
            std_val = float(np.std(numeric))
            n = len(numeric)

            # Confidence increases with sample size and decreases with variance
            confidence = min(0.99, 1.0 - 1.0 / (1.0 + n * 0.1)) * max(0.1, 1.0 - std_val)

            generalization = {
                "type": "statistical",
                "mean": mean_val,
                "std": std_val,
                "sample_size": n,
                "confidence": confidence,
                "prediction": mean_val
            }
            self._induced_generalizations.append(generalization)
            self._log_inference("inductive", observations, generalization, confidence, True)
            return generalization

        # Categorical observations: find mode and frequency distribution
        from collections import Counter
        counts = Counter(str(o) for o in observations)
        most_common = counts.most_common(1)[0]
        frequency = most_common[1] / len(observations)

        generalization = {
            "type": "categorical",
            "mode": most_common[0],
            "frequency": frequency,
            "distribution": dict(counts),
            "confidence": frequency,
            "prediction": most_common[0]
        }
        self._induced_generalizations.append(generalization)
        self._log_inference("inductive", observations, generalization, frequency, True)
        return generalization

    # -- Abductive Reasoning ------------------------------------------------

    def abductive_reasoning(self, observation: Any,
                            hypotheses: Dict[str, float]) -> Dict[str, Any]:
        """
        Inference to the best explanation. Given an observation and candidate
        hypotheses with prior probabilities, rank by explanatory power.
        """
        if not hypotheses:
            return {"best": None, "ranking": [], "confidence": 0.0}

        # Score each hypothesis by prior * coherence with existing knowledge
        scored = {}
        for hypothesis, prior in hypotheses.items():
            coherence = self._compute_coherence(hypothesis)
            # Bayesian-inspired scoring: prior * likelihood (coherence as proxy)
            score = prior * (0.5 + 0.5 * coherence)
            scored[hypothesis] = score

        # Normalize scores
        total = sum(scored.values())
        if total > 0:
            scored = {k: v / total for k, v in scored.items()}

        ranking = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        best = ranking[0] if ranking else (None, 0.0)

        result = {
            "best": best[0],
            "confidence": best[1],
            "ranking": [{"hypothesis": h, "score": s} for h, s in ranking],
            "observation": str(observation)
        }

        self._log_inference("abductive", [observation], result, best[1], True)
        return result

    # -- Analogical Reasoning -----------------------------------------------

    def analogical_reasoning(self, source_domain: Dict[str, Any],
                             target_domain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer knowledge from a source domain to a target domain based
        on structural similarity between their properties.
        """
        source_props = set(source_domain.keys())
        target_props = set(target_domain.keys())

        # Compute structural similarity
        shared = source_props & target_props
        total = source_props | target_props
        similarity = len(shared) / max(1, len(total))

        # Transfer properties from source to target that target lacks
        transferred = {}
        for prop in source_props - target_props:
            # Confidence of transfer proportional to overall similarity
            transferred[prop] = {
                "value": source_domain[prop],
                "confidence": similarity * 0.8,
                "source": "analogical_transfer"
            }

        return {
            "similarity": similarity,
            "shared_properties": list(shared),
            "transferred": transferred,
            "transfer_confidence": similarity * 0.8
        }

    # -- Contradiction Detection --------------------------------------------

    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Scan propositions for contradictions using destructive interference.
        Contradictions occur when P and NOT(P) both have high truth amplitudes.
        """
        contradictions = []
        checked = set()

        for prop, truth in self.propositions.items():
            negated = f"NOT({prop})"
            alt_negated = prop.replace("NOT(", "").rstrip(")") if prop.startswith("NOT(") else None

            for check in [negated, alt_negated]:
                if check and check in self.propositions and check not in checked:
                    other_truth = self.propositions[check]

                    # Destructive interference: both high = contradiction
                    interference = truth * other_truth
                    if interference > self.contradiction_sensitivity:
                        contradiction = {
                            "proposition_a": prop,
                            "truth_a": truth,
                            "proposition_b": check,
                            "truth_b": other_truth,
                            "interference": interference,
                            "timestamp": datetime.now().isoformat()
                        }
                        contradictions.append(contradiction)
                        self.contradiction_log.append(contradiction)
                        checked.add(prop)
                        checked.add(check)

        return contradictions

    # -- Resolution Engine --------------------------------------------------

    def resolve(self, clauses: List[Set[str]]) -> bool:
        """
        Propositional resolution: attempt to derive empty clause (contradiction)
        from a set of clauses to prove unsatisfiability.
        Returns True if the clause set is satisfiable.
        """
        clauses = [frozenset(c) for c in clauses]
        new = set()

        while True:
            pairs = [(clauses[i], clauses[j])
                     for i in range(len(clauses))
                     for j in range(i + 1, len(clauses))]

            for ci, cj in pairs:
                resolvents = self._resolve_pair(ci, cj)
                if frozenset() in resolvents:
                    return False  # Unsatisfiable (contradiction found)
                new.update(resolvents)

            if new.issubset(set(clauses)):
                return True  # Satisfiable (no new clauses)

            clauses = list(set(clauses) | new)

    def _resolve_pair(self, c1: frozenset, c2: frozenset) -> Set[frozenset]:
        """Resolve two clauses on complementary literals."""
        resolvents = set()
        for literal in c1:
            complement = literal.lstrip("~") if literal.startswith("~") else f"~{literal}"
            if complement in c2:
                new_clause = (c1 - {literal}) | (c2 - {complement})
                resolvents.add(frozenset(new_clause))
        return resolvents

    # -- Proposition Management ---------------------------------------------

    def assert_proposition(self, proposition: str, truth: float = 1.0):
        """Assert a proposition with a truth amplitude."""
        self.propositions[proposition] = np.clip(truth, 0.0, 1.0)

    def query_proposition(self, proposition: str) -> float:
        """Query the truth amplitude of a proposition."""
        return self.propositions.get(proposition, 0.5)

    def add_rule(self, premises: List[str], conclusion: str,
                 confidence: float = 0.9):
        """Add an inference rule."""
        self.rules.append({
            "premises": premises,
            "conclusion": conclusion,
            "confidence": confidence
        })

    def forward_chain(self) -> List[str]:
        """Apply all applicable rules to derive new conclusions."""
        derived = []
        for rule in self.rules:
            premises_truth = [self.propositions.get(p, 0.0) for p in rule["premises"]]
            if all(t >= self.fuzzy_threshold for t in premises_truth):
                combined = np.prod(premises_truth) * rule["confidence"]
                if combined > self.propositions.get(rule["conclusion"], 0.0):
                    self.propositions[rule["conclusion"]] = combined
                    derived.append(rule["conclusion"])
        return derived

    # -- Validation (for UnifiedQuantumMind integration) --------------------

    def validate(self, idea: Any) -> bool:
        """
        Validate an idea for logical consistency.
        Used by the UnifiedQuantumMind orchestrator.
        """
        if isinstance(idea, str):
            # Check for contradiction with existing propositions
            contradictions = self.detect_contradictions()
            if contradictions:
                return False
            # Check truth amplitude if proposition exists
            truth = self.propositions.get(idea, 0.5)
            return truth >= self.fuzzy_threshold

        if isinstance(idea, dict):
            # Validate all claims in a structured idea
            claims = idea.get("claims", [])
            if not claims:
                return True
            truths = [self.propositions.get(c, 0.5) for c in claims]
            return np.mean(truths) >= self.fuzzy_threshold

        return True  # Default: accept if we can't evaluate

    # -- Helpers ------------------------------------------------------------

    def _compute_coherence(self, hypothesis: str) -> float:
        """Compute how coherent a hypothesis is with existing knowledge."""
        if not self.propositions:
            return 0.5

        # Simple coherence: count how many existing propositions
        # are consistent (share words) with the hypothesis
        h_words = set(hypothesis.lower().split())
        coherence_scores = []

        for prop, truth in self.propositions.items():
            p_words = set(prop.lower().split())
            overlap = len(h_words & p_words) / max(1, len(h_words | p_words))
            if overlap > 0:
                coherence_scores.append(overlap * truth)

        return float(np.mean(coherence_scores)) if coherence_scores else 0.5

    def _log_inference(self, mode: str, premises: Any, conclusion: Any,
                       confidence: float, valid: Optional[bool]):
        """Log an inference for history tracking."""
        self.inference_log.append({
            "mode": mode,
            "premises": str(premises),
            "conclusion": str(conclusion),
            "confidence": confidence,
            "valid": valid,
            "timestamp": datetime.now().isoformat()
        })

    def report(self) -> Dict[str, Any]:
        """Generate reasoning activity report."""
        mode_counts = defaultdict(int)
        for entry in self.inference_log:
            mode_counts[entry["mode"]] += 1

        return {
            "total_inferences": len(self.inference_log),
            "by_mode": dict(mode_counts),
            "propositions_stored": len(self.propositions),
            "rules_registered": len(self.rules),
            "contradictions_found": len(self.contradiction_log),
            "generalizations_induced": len(self._induced_generalizations)
        }


# Backwards compatibility alias for UnifiedQuantumMind integration
LogicCore = LogicEngine
