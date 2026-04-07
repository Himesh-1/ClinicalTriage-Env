"""
grader.py — Per-task grader functions that return float 0.0–1.0.

Each grader accepts the raw action dict (as sent by the agent) and
the session state, and returns a scalar reward.
"""

from typing import Any, Dict, List
from models import TriageAction
from environment import ESIGrader, kendall_tau_reward


# ─────────────────────────────────────────────────────────────────────────────
#  Task 1 — Easy: single patient
# ─────────────────────────────────────────────────────────────────────────────

def grade_single_patient(
    action: TriageAction,
    ground_truth_esi: int,
    step_count: int = 1,
) -> float:
    """
    Grade a single-patient triage action.
    Returns float in [0.0, 1.0].
    """
    return ESIGrader.grade(action, ground_truth_esi, step_count)


# ─────────────────────────────────────────────────────────────────────────────
#  Task 2 — Medium: concurrent patients ranking
# ─────────────────────────────────────────────────────────────────────────────

def grade_concurrent_ranking(
    agent_ranking: List[str],
    ground_truth_ranking: List[str],
) -> float:
    """
    Grade the agent's ranking of 5 concurrent patients using
    Kendall Tau rank correlation.

    agent_ranking: list of patient_id strings in agent's priority order
    ground_truth_ranking: correct priority order

    Returns float in [0.0, 1.0].
    """
    return kendall_tau_reward(agent_ranking, ground_truth_ranking)


# ─────────────────────────────────────────────────────────────────────────────
#  Task 3 — Hard: incomplete vitals
# ─────────────────────────────────────────────────────────────────────────────

def grade_incomplete_vitals(
    action: TriageAction,
    ground_truth_esi: int,
    step_count: int,
) -> float:
    """
    Grade a triage action with step-decay penalty.
    step_count includes any 'request_vitals' steps taken before final action.

    Reward decays by 0.05 per step beyond step 1.
    Returns float in [0.0, 1.0].
    """
    return ESIGrader.grade(action, ground_truth_esi, step_count)


# ─────────────────────────────────────────────────────────────────────────────
#  Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def grade(task_name: str, payload: Dict[str, Any]) -> float:
    """
    Generic grader dispatcher.

    payload keys:
      - action (TriageAction)  — for easy and hard tasks
      - ground_truth_esi (int) — for easy and hard tasks
      - step_count (int)       — optional, defaults to 1
      - agent_ranking (list)   — for medium task
      - ground_truth_ranking (list) — for medium task
    """
    if task_name == "single_patient_easy":
        return grade_single_patient(
            action=payload["action"],
            ground_truth_esi=payload["ground_truth_esi"],
            step_count=payload.get("step_count", 1),
        )
    elif task_name == "concurrent_patients_medium":
        return grade_concurrent_ranking(
            agent_ranking=payload["agent_ranking"],
            ground_truth_ranking=payload["ground_truth_ranking"],
        )
    elif task_name == "incomplete_vitals_hard":
        return grade_incomplete_vitals(
            action=payload["action"],
            ground_truth_esi=payload["ground_truth_esi"],
            step_count=payload.get("step_count", 1),
        )
    else:
        raise ValueError(f"Unknown task for grader: '{task_name}'")
