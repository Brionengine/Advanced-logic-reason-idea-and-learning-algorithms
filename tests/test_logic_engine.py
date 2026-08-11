"""
Tests for the Logic Engine (QELR).

Covers the truth-maintenance layer (justification tracking, explanation,
cascading retraction, fixpoint chaining) alongside the pre-existing reasoning
modes, so a regression in either surfaces here.
"""

import pytest

from logic_engine import LogicEngine, LogicCore


@pytest.fixture
def engine():
    return LogicEngine()


# -- Fixpoint forward chaining ---------------------------------------------


def test_single_pass_misses_chain_under_unfavourable_rule_order(engine):
    """A one-pass chain only fires rules whose premises are already known."""
    engine.assert_proposition("A", 1.0)
    # C's rule is evaluated before the rule that establishes its premise B.
    engine.add_rule(["B"], "C", 0.9)
    engine.add_rule(["A"], "B", 0.9)

    assert engine.forward_chain() == ["B"]


def test_fixpoint_completes_the_chain(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["B"], "C", 0.9)
    engine.add_rule(["A"], "B", 0.9)

    result = engine.forward_chain_to_fixpoint()

    assert set(result["unique_derived"]) == {"B", "C"}
    assert result["converged"] is True
    assert result["iterations"] > 1


def test_fixpoint_converges_immediately_when_nothing_fires(engine):
    engine.add_rule(["missing"], "X", 0.9)

    result = engine.forward_chain_to_fixpoint()

    assert result["unique_derived"] == []
    assert result["converged"] is True


def test_fixpoint_respects_iteration_cap(engine):
    """A cap of 1 cannot converge on a two-link chain."""
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["B"], "C", 0.9)
    engine.add_rule(["A"], "B", 0.9)

    result = engine.forward_chain_to_fixpoint(max_iterations=1)

    assert result["iterations"] == 1
    assert result["converged"] is False


# -- Justification tracking and explanation --------------------------------


def test_derived_proposition_records_its_support(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()

    supports = engine.justifications["B"]
    assert len(supports) == 1
    assert supports[0]["premises"] == ["A"]
    assert supports[0]["mode"] == "forward_chain"


def test_asserted_proposition_has_no_justification(engine):
    engine.assert_proposition("A", 1.0)
    assert engine.explain("A")["basis"] == "asserted"


def test_unknown_proposition_reports_unknown(engine):
    assert engine.explain("never_seen")["basis"] == "unknown"


def test_explain_walks_back_to_premises(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.add_rule(["B"], "C", 0.9)
    engine.forward_chain_to_fixpoint()

    tree = engine.explain("C")

    assert tree["basis"] == "derived"
    b_node = tree["supports"][0]["premises"][0]
    assert b_node["proposition"] == "B"
    a_node = b_node["supports"][0]["premises"][0]
    assert a_node["proposition"] == "A"
    assert a_node["basis"] == "asserted"


def test_explain_breaks_cycles_rather_than_hanging(engine):
    """Mutually supporting beliefs must not send the walk into recursion."""
    engine.assert_proposition("P", 0.9)
    engine.assert_proposition("Q", 0.9)
    engine._record_justification("P", ["Q"], "forward_chain", 0.9)
    engine._record_justification("Q", ["P"], "forward_chain", 0.9)

    text = engine.explain_text("P")

    assert "circular" in text


def test_explain_respects_depth_limit(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.add_rule(["B"], "C", 0.9)
    engine.forward_chain_to_fixpoint()

    tree = engine.explain("C", max_depth=1)
    nested = tree["supports"][0]["premises"][0]

    assert nested["basis"] == "depth_limit"


def test_explain_text_renders_proof_sketch(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()

    text = engine.explain_text("B")

    assert "B" in text
    assert "forward_chain" in text
    assert "asserted" in text


def test_repeated_derivation_does_not_duplicate_support(engine):
    """Re-running the same rule is not a second independent justification."""
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()
    engine.propositions["B"] = 0.0  # force the rule to fire again
    engine.forward_chain()

    assert len(engine.justifications["B"]) == 1


# -- Retraction -------------------------------------------------------------


def test_retract_cascades_to_dependent_conclusions(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.add_rule(["B"], "C", 0.9)
    engine.forward_chain_to_fixpoint()

    result = engine.retract("A")

    assert set(result["removed"]) == {"A", "B", "C"}
    assert engine.propositions == {}


def test_retract_without_cascade_leaves_dependents(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()

    engine.retract("A", cascade=False)

    assert "A" not in engine.propositions
    assert "B" in engine.propositions


def test_over_determined_conclusion_survives_partial_retraction(engine):
    """A conclusion with a second independent support must not be withdrawn."""
    engine.assert_proposition("A", 1.0)
    engine.assert_proposition("X", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.add_rule(["X"], "B", 0.9)
    engine.forward_chain_to_fixpoint()
    assert len(engine.justifications["B"]) == 2

    engine.retract("A")

    assert "B" in engine.propositions
    assert len(engine.justifications["B"]) == 1


def test_retracting_unknown_proposition_is_harmless(engine):
    result = engine.retract("never_existed")
    assert result["removed"] == []


# -- Resolution -------------------------------------------------------------


def test_resolve_detects_contradiction(engine):
    assert engine.resolve([{"p"}, {"~p"}]) is False


def test_resolve_accepts_satisfiable_clauses(engine):
    assert engine.resolve([{"p", "q"}, {"~p", "q"}]) is True


def test_resolve_finds_indirect_contradiction(engine):
    # p, ~p or q, ~q  ->  empty clause
    assert engine.resolve([{"p"}, {"~p", "q"}, {"~q"}]) is False


def test_resolve_terminates_on_bounded_budget(engine):
    """A tight clause budget must return rather than expand indefinitely."""
    clauses = [{f"v{i}", f"~v{i + 1}"} for i in range(12)]
    assert engine.resolve(clauses, max_rounds=2, max_clauses=8) is True


# -- Pre-existing reasoning modes -------------------------------------------


def test_deductive_reasoning_accepts_strong_premises(engine):
    engine.assert_proposition("P", 1.0)
    engine.assert_proposition("Q", 1.0)
    assert engine.deductive_reasoning(["P", "Q"], "R") == "R"


def test_deductive_reasoning_negates_on_weak_premises(engine):
    engine.assert_proposition("P", 0.0)
    assert engine.deductive_reasoning(["P"], "R").startswith("Negated")


def test_deductive_reasoning_without_premises_is_undetermined(engine):
    assert engine.deductive_reasoning([], "R") == "Undetermined"


def test_modus_ponens_derives_consequent(engine):
    engine.assert_proposition("P", 1.0)
    assert engine.modus_ponens("P", "Q") == "Q"
    assert engine.propositions["Q"] > 0


def test_modus_ponens_declines_when_antecedent_weak(engine):
    engine.assert_proposition("P", 0.1)
    assert engine.modus_ponens("P", "Q") is None


def test_modus_tollens_negates_antecedent(engine):
    engine.assert_proposition("Q", 0.0)
    assert engine.modus_tollens("P", "Q") == "NOT(P)"


def test_syllogism_links_matching_terms(engine):
    result = engine.syllogism(("men", "mortal"), ("mortal", "finite"))
    assert result == "All men are finite"


def test_syllogism_rejects_unlinked_terms(engine):
    assert engine.syllogism(("men", "mortal"), ("cats", "furry")) is None


def test_inductive_reasoning_summarises_numeric_observations(engine):
    result = engine.inductive_reasoning([2, 2, 2, 2])
    assert result["type"] == "statistical"
    assert result["mean"] == pytest.approx(2.0)


def test_inductive_reasoning_summarises_categorical_observations(engine):
    result = engine.inductive_reasoning(["red", "red", "blue"])
    assert result["type"] == "categorical"
    assert result["mode"] == "red"


def test_inductive_reasoning_without_observations(engine):
    assert engine.inductive_reasoning([]) is None


def test_abductive_reasoning_ranks_hypotheses(engine):
    result = engine.abductive_reasoning("symptom", {"flu": 0.7, "cold": 0.3})
    assert result["best"] == "flu"
    assert len(result["ranking"]) == 2


def test_abductive_reasoning_without_hypotheses(engine):
    assert engine.abductive_reasoning("symptom", {})["best"] is None


def test_analogical_reasoning_transfers_unshared_properties(engine):
    result = engine.analogical_reasoning(
        {"shape": "round", "colour": "red", "orbit": True},
        {"shape": "round", "colour": "blue"},
    )
    assert "orbit" in result["transferred"]
    assert result["similarity"] > 0


def test_contradiction_detection_flags_opposing_beliefs(engine):
    engine.assert_proposition("P", 1.0)
    engine.assert_proposition("NOT(P)", 1.0)
    assert len(engine.detect_contradictions()) == 1


def test_no_contradiction_when_beliefs_agree(engine):
    engine.assert_proposition("P", 1.0)
    engine.assert_proposition("Q", 1.0)
    assert engine.detect_contradictions() == []


def test_assert_clamps_truth_to_unit_interval(engine):
    engine.assert_proposition("P", 5.0)
    engine.assert_proposition("Q", -5.0)
    assert engine.propositions["P"] == 1.0
    assert engine.propositions["Q"] == 0.0


def test_query_unknown_proposition_returns_uncertain(engine):
    assert engine.query_proposition("unknown") == 0.5


# -- Reporting --------------------------------------------------------------


def test_justification_report_separates_derived_from_asserted(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()

    report = engine.justification_report()

    assert report["asserted_propositions"] == ["A"]
    assert report["derived_propositions"] == ["B"]


def test_report_counts_justified_conclusions(engine):
    engine.assert_proposition("A", 1.0)
    engine.add_rule(["A"], "B", 0.9)
    engine.forward_chain()

    assert engine.report()["justified_conclusions"] == 1


def test_validate_accepts_structured_idea(engine):
    engine.assert_proposition("claim1", 1.0)
    assert engine.validate({"claims": ["claim1"]}) is True


def test_logic_core_alias_points_at_engine():
    assert LogicCore is LogicEngine
