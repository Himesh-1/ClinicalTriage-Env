# tests/test_rubrics.py
"""Unit tests for rubric components. No server, no LLM, no network needed."""
import pytest
from models import TriageAction, PatientObservation
from rubrics import (
    ExactMatchRubric, ProximityRubric, ReasoningRubric,
    KendallTauRubric, ClinicalTriageRubric,
)


def test_exact_match_hit():
    assert ExactMatchRubric().grade(2, 2) == 0.6


def test_exact_match_miss():
    assert ExactMatchRubric().grade(3, 1) == 0.0


def test_proximity_off_by_one():
    assert ProximityRubric().grade(2, 3, exact_matched=False) == 0.3


def test_proximity_off_by_two():
    assert ProximityRubric().grade(1, 3, exact_matched=False) == 0.0


def test_proximity_suppressed_on_exact():
    assert ProximityRubric().grade(2, 2, exact_matched=True) == 0.0


def test_reasoning_heuristic_in_range():
    r = ReasoningRubric()
    score = r.grade("ESI-1: HR 130, BP 80/50, SpO2 88% — critical vitals.", skip_llm=True)
    assert 0.0 <= score <= r.weight


def test_reasoning_empty_string():
    r = ReasoningRubric()
    score = r.grade("", skip_llm=True)
    assert score == 0.0


def test_kendall_tau_perfect():
    r = KendallTauRubric()
    gt = ["P001", "P002", "P003"]
    agent = ["P001", "P002", "P003"]
    assert r.grade(agent, gt) == 1.0


def test_kendall_tau_reversed():
    r = KendallTauRubric()
    gt = ["P001", "P002", "P003"]
    assert r.grade(["P003", "P002", "P001"], gt) == 0.0


def test_kendall_tau_empty():
    assert KendallTauRubric().grade([], []) == 0.0


def test_reward_in_unit_interval():
    rubric = ClinicalTriageRubric()
    action = TriageAction(
        triage_level=1,
        reasoning="Critical: BP 80/50, SpO2 88%, HR 130. ESI-1.",
        confidence=0.95,
    )
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=1, skip_llm=True)
    assert 0.0 <= reward <= 1.0


def test_reward_never_exceeds_one():
    rubric = ClinicalTriageRubric()
    action = TriageAction(
        triage_level=1,
        reasoning="x" * 500,  # very long reasoning
        confidence=1.0,
    )
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=1, skip_llm=True)
    assert reward <= 1.0


def test_reward_never_below_zero():
    rubric = ClinicalTriageRubric()
    action = TriageAction(triage_level=5, reasoning="", confidence=0.0)
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=10, skip_llm=True)
    assert reward >= 0.0
