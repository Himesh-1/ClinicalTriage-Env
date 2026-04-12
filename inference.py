# inference.py
"""
ClinicalTriage-Env — Hackathon Inference Script

MANDATORY env vars:
    API_BASE_URL    LLM API endpoint            (default provided)
    MODEL_NAME      Model identifier             (default provided)
    HF_TOKEN        HuggingFace/API key          (required, no default)
    ENV_BASE_URL    Running env server URL       (default: http://localhost:7860)
    LOCAL_IMAGE_NAME Docker image name           (default: clinicaltriage-env:latest)

STDOUT FORMAT (exactly):
    [START] task=<task> env=clinicaltriage model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import json
import os
import sys
import uuid
from typing import Optional

from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME: str = os.getenv(
    "MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME: str = os.getenv(
    "LOCAL_IMAGE_NAME", "clinicaltriage-env:latest"
)

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client ─────────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Stdout loggers ────────────────────────────────────────────────────────────

def _log_start(task: str, model: str) -> None:
    print(
        f"[START] task={task} env=clinicaltriage model={model}",
        flush=True,
    )


def _log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    # Sanitise action: collapse newlines, truncate to 120 chars
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def _log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── LLM agent ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a trained emergency triage nurse. "
    "Assess the patient and assign an ESI level (1=Critical, 5=Non-Urgent). "
    "For multiple patients, rank them by priority (most critical first). "
    "If triage_level is 0, you are requesting more vitals information."
)


def _get_action(obs: dict) -> dict:
    """
    Call the LLM and return a parsed TriageAction dict.
    Falls back to a safe default on any exception.
    """
    prompt = (
        "Patient data:\n"
        f"{json.dumps(obs, indent=2)}\n\n"
        "Respond ONLY with valid JSON (no markdown, no explanation):\n"
        '{"triage_level": <int 1-5>, '
        '"reasoning": "<clinical justification>", '
        '"confidence": <float 0.0-1.0>}'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if model wraps output
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return {
            "triage_level": 3,
            "reasoning": "Fallback: LLM unavailable, defaulting to ESI-3 urgent.",
            "confidence": 0.1,
        }

# ── Environment HTTP helpers ──────────────────────────────────────────────────

def _reset(task_name: str, session_id: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _step(action: dict, session_id: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task_name: str) -> None:
    """
    Run one episode of the given task.
    Emits [START], one [STEP] per step, then [END].
    [END] is guaranteed even on exception via finally.
    """
    _log_start(task=task_name, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken: int = 0
    success: bool = False
    score: float = 0.0

    session_id = f"inference_{task_name}_{uuid.uuid4().hex[:8]}"
    try:
        reset_data = _reset(task_name, session_id)

        # Determine observation shape
        if "patients" in reset_data:
            # Medium task: list of patients
            obs = {"patients": reset_data["patients"]}
        else:
            obs = reset_data.get("observation", reset_data)

        done = False
        max_steps = 10  # safety cap — tasks define their own, but cap defensively

        while not done and steps_taken < max_steps:
            steps_taken += 1
            action_dict = _get_action(obs)
            action_str = json.dumps(action_dict)

            try:
                result = _step(action_dict, session_id)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", True))
                error = result.get("info", {}).get("error", None)

                # If the server revealed vitals (hard task, triage_level=0),
                # update obs and do not count this as a scored step
                if result.get("observation") and not result.get("done"):
                    obs = result["observation"]
                    _log_step(
                        step=steps_taken,
                        action=action_str,
                        reward=0.0,
                        done=False,
                        error=error,
                    )
                    continue

                rewards.append(reward)
                _log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )

            except requests.HTTPError as exc:
                error_msg = str(exc)
                rewards.append(0.0)
                _log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=error_msg,
                )
                done = True

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {error_msg}", file=sys.stderr, flush=True)
        if not rewards:
            rewards = [0.0]
            steps_taken = max(steps_taken, 1)
        _log_step(
            step=steps_taken,
            action="null",
            reward=0.0,
            done=True,
            error=error_msg,
        )
        score = 0.0
        success = False

    finally:
        _log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards if rewards else [0.0],
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TASKS = [
        "single_patient_easy",
        "concurrent_patients_medium",
        "incomplete_vitals_hard",
    ]
    for task in TASKS:
        run_task(task)
