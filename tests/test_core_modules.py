"""
Smoke tests for the core reasoning modules.

These guard the cheapest failure mode there is: a module that no longer
imports, or whose primary class no longer instantiates. Nothing here asserts
deep behaviour — that belongs in the per-module suites — but a break in any of
these means the package is unusable regardless of what else passes.
"""

import importlib

import pytest

CORE_MODULES = [
    "logic_engine",
    "meta_reasoning",
    "knowledge_base",
    "epistemic_confidence",
    "probabilistic_reasoning",
    "abductive_reasoning",
    "experience_replay",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_logic_engine_instantiates():
    from logic_engine import LogicEngine

    engine = LogicEngine()
    assert engine.propositions == {}
    assert engine.report()["total_inferences"] == 0


def test_engine_accepts_custom_thresholds():
    from logic_engine import LogicEngine

    engine = LogicEngine(fuzzy_threshold=0.7, contradiction_sensitivity=0.6)
    assert engine.fuzzy_threshold == 0.7
    assert engine.contradiction_sensitivity == 0.6


def test_reasoning_pipeline_end_to_end():
    """Assert facts, chain rules, explain the result, then withdraw the premise."""
    from logic_engine import LogicEngine

    engine = LogicEngine()
    engine.assert_proposition("sensor_online", 1.0)
    engine.add_rule(["sensor_online"], "telemetry_valid", 0.95)
    engine.add_rule(["telemetry_valid"], "navigation_safe", 0.9)

    result = engine.forward_chain_to_fixpoint()
    assert "navigation_safe" in result["unique_derived"]

    explanation = engine.explain_text("navigation_safe")
    assert "sensor_online" in explanation

    engine.retract("sensor_online")
    assert "navigation_safe" not in engine.propositions
