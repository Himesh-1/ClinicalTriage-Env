"""
server.py — FastAPI application (OpenEnv server)

Endpoints:
  POST /reset  → accepts optional task_name, returns PatientObservation
  POST /step   → accepts TriageAction, returns StepResult
  GET  /state  → returns EnvironmentState
  GET  /health → returns {"status": "ok"}
  GET  /tasks  → returns list of available tasks

Session state is stored in an in-memory dict keyed by session_id (UUID).
"""

import re
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from models import (
    PatientObservation,
    TriageAction,
    EnvironmentState,
    StepResult,
    HealthResponse,
    TaskInfo,
    ResetRequest,
    StepRequest,
)
from environment import PatientSimulator, ESIGrader, kendall_tau_reward
from tasks import get_task, list_tasks
import grader as grader_module

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClinicalTriage-Env",
    description=(
        "OpenEnv RL environment for AI-powered emergency room patient triage "
        "using the ESI (Emergency Severity Index) scale. "
        "Simulates Indian ER scenarios with 3 difficulty levels."
    ),
    version="1.0.0",
)

# ─── In-memory session store ──────────────────────────────────────────────────
# Structure per session:
# {
#   "task_name": str,
#   "step_count": int,
#   "done": bool,
#   "cumulative_reward": float,
#   "current_observations": [PatientObservation, ...],
#   "ground_truth_map": {patient_id: esi_int},
#   "raw_profiles": [dict, ...],
#   "ground_truth_ranking": [patient_id, ...]  # medium task only
#   "set_index": int  # which patient set (medium task)
# }
_sessions: Dict[str, Dict[str, Any]] = {}

# Per-task episode counter for cycling
_episode_counters: Dict[str, int] = {}

DEFAULT_SESSION_ID = "default"


def _get_or_create_session(session_id: Optional[str]) -> str:
    sid = session_id or DEFAULT_SESSION_ID
    if sid not in _sessions:
        _sessions[sid] = {
            "task_name": "single_patient_easy",
            "step_count": 0,
            "done": True,
            "cumulative_reward": 0.0,
            "current_observations": [],
            "ground_truth_map": {},
            "raw_profiles": [],
            "ground_truth_ranking": [],
            "set_index": 0,
        }
    return sid


def _build_state(sid: str) -> EnvironmentState:
    s = _sessions[sid]
    return EnvironmentState(
        task_name=s["task_name"],
        step_count=s["step_count"],
        current_patients=s["current_observations"],
        done=s["done"],
        cumulative_reward=s["cumulative_reward"],
    )


def _next_episode_index(task_name: str) -> int:
    """Return and increment the episode counter for a task."""
    if task_name not in _episode_counters:
        _episode_counters[task_name] = 0
    idx = _episode_counters[task_name]
    _episode_counters[task_name] += 1
    return idx


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — used by HF Spaces auto-ping."""
    return HealthResponse(status="ok")


@app.get("/tasks", tags=["system"])
def tasks():
    """List all available task names and descriptions."""
    return list_tasks()


@app.post("/reset", response_model=PatientObservation, tags=["env"])
def reset(body: ResetRequest = ResetRequest()):
    """
    Reset the environment and return the first patient observation.
    Each call cycles to a different patient profile for variety.
    """
    task_name = body.task_name or "single_patient_easy"
    sid = _get_or_create_session(body.session_id)

    try:
        task_cfg = get_task(task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get episode index for cycling through profiles
    ep_idx = _next_episode_index(f"{sid}_{task_name}")

    # ── Build session state ──────────────────────────────────────────────────
    ground_truth_map: Dict[str, int] = {}
    raw_profiles_list = []
    observations = []
    ground_truth_ranking = []
    set_index = 0

    if task_name == "concurrent_patients_medium":
        # Cycle through patient sets
        sets = task_cfg.get("patient_sets", [])
        if sets:
            set_index = ep_idx % len(sets)
        all_patients = PatientSimulator.generate_all(task_name, set_index=set_index)
        for obs, gt_esi, raw in all_patients:
            observations.append(obs)
            ground_truth_map[obs.patient_id] = gt_esi
            raw_profiles_list.append(raw)
        ground_truth_ranking = PatientSimulator.get_medium_ground_truth(
            task_name, set_index=set_index
        )
    else:
        # Easy / Hard: cycle through profiles
        profiles = task_cfg["patient_profiles"]
        profile_idx = ep_idx % len(profiles)
        obs, gt_esi, raw = PatientSimulator.generate(task_name, profile_index=profile_idx)
        observations.append(obs)
        ground_truth_map[obs.patient_id] = gt_esi
        raw_profiles_list.append(raw)

    _sessions[sid] = {
        "task_name": task_name,
        "step_count": 0,
        "done": False,
        "cumulative_reward": 0.0,
        "current_observations": observations,
        "ground_truth_map": ground_truth_map,
        "raw_profiles": raw_profiles_list,
        "ground_truth_ranking": ground_truth_ranking,
        "set_index": set_index,
    }

    # Return the first (or only) observation
    return observations[0]


@app.post("/step", response_model=StepResult, tags=["env"])
def step(body: StepRequest):
    """
    Submit a TriageAction and receive the step result.

    For Easy/Hard: action applies to the current single patient.
    For Medium: action.reasoning should contain a JSON ranking of patient_ids.
    For Hard: if action.triage_level == 0, treat it as a 'request_vitals' action.
    """
    sid = _get_or_create_session(body.session_id)
    s = _sessions[sid]

    if s["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    action = body.action
    task_name = s["task_name"]
    s["step_count"] += 1
    step_count = s["step_count"]

    reward = 0.0
    done = False
    info: Dict[str, Any] = {"task": task_name, "step": step_count}

    # ── Easy ─────────────────────────────────────────────────────────────────
    if task_name == "single_patient_easy":
        obs = s["current_observations"][0]
        gt_esi = s["ground_truth_map"][obs.patient_id]
        reward = grader_module.grade(
            task_name,
            {"action": action, "ground_truth_esi": gt_esi, "step_count": step_count},
        )
        done = True
        info["ground_truth_esi"] = gt_esi
        info["agent_level"] = action.triage_level
        next_obs = obs

    # ── Medium ────────────────────────────────────────────────────────────────
    elif task_name == "concurrent_patients_medium":
        # Agent encodes their ranking as a JSON list in action.reasoning
        agent_ranking = _extract_medium_ranking(action.reasoning, s["ground_truth_ranking"])

        reward = grader_module.grade(
            task_name,
            {
                "agent_ranking": agent_ranking,
                "ground_truth_ranking": s["ground_truth_ranking"],
            },
        )
        done = True
        info["agent_ranking"] = agent_ranking
        info["ground_truth_ranking"] = s["ground_truth_ranking"]
        next_obs = s["current_observations"][0]

    # ── Hard ──────────────────────────────────────────────────────────────────
    elif task_name == "incomplete_vitals_hard":
        obs = s["current_observations"][0]
        raw = s["raw_profiles"][0]
        task_cfg = get_task(task_name)

        # triage_level == 0 means "request_vitals"
        if action.triage_level == 0:
            # Reveal complete vitals
            updated_obs = PatientSimulator.reveal_vitals(raw)
            s["current_observations"][0] = updated_obs
            reward = 0.0  # No reward for requesting vitals
            done = False
            info["action_type"] = "request_vitals"
            info["vitals_revealed"] = True
            next_obs = updated_obs
        else:
            # Final triage decision
            gt_esi = s["ground_truth_map"][obs.patient_id]
            reward = grader_module.grade(
                task_name,
                {
                    "action": action,
                    "ground_truth_esi": gt_esi,
                    "step_count": step_count,
                },
            )
            done = True
            info["ground_truth_esi"] = gt_esi
            info["agent_level"] = action.triage_level
            info["steps_used"] = step_count
            next_obs = obs

        # Force done if exceeded max steps
        if step_count >= task_cfg.get("max_steps", 3):
            done = True

    else:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")

    s["cumulative_reward"] += reward
    s["done"] = done

    return StepResult(
        observation=next_obs,
        reward=reward,
        done=done,
        info=info,
    )


def _extract_medium_ranking(reasoning: str, ground_truth_ranking: list) -> list:
    """
    Extract patient IDs from reasoning in the ORDER THEY APPEAR.
    First tries JSON parsing, then falls back to regex extraction.

    BUG FIX: The old code searched in ground-truth order, which always
    produced the correct ranking. Now we search in order of appearance.
    """
    # Attempt 1: Try direct JSON parse
    try:
        parsed = json.loads(reasoning)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: Extract IDs in order they appear via regex
    # Pattern matches common patient ID formats: M001, MA01, MB02, etc.
    valid_ids = set(ground_truth_ranking)
    found = []
    for match in re.finditer(r'[A-Z]{1,2}\d{2,3}', reasoning):
        pid = match.group()
        if pid in valid_ids and pid not in found:
            found.append(pid)

    if found:
        return found

    # Attempt 3: Last resort — search text for each ID in appearance order
    positions = []
    for pid in ground_truth_ranking:
        pos = reasoning.find(pid)
        if pos != -1:
            positions.append((pos, pid))

    if positions:
        positions.sort(key=lambda x: x[0])
        return [pid for _, pid in positions]

    return []


@app.get("/state", response_model=EnvironmentState, tags=["env"])
def state(session_id: Optional[str] = Query(default=None)):
    """Return the current environment state."""
    sid = _get_or_create_session(session_id)
    return _build_state(sid)


# ─────────────────────────────────────────────────────────────────────────────
#  Exception handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
