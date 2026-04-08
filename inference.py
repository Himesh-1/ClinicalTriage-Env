"""
inference.py — Baseline LLM agent for ClinicalTriage-Env

CRITICAL REQUIREMENTS (from spec):
  - Must be in root directory
  - Must use OpenAI client (from openai import OpenAI)
  - Must use API_BASE_URL, MODEL_NAME, HF_TOKEN env vars
  - Must output EXACT [START]/[STEP]/[END] log format
  - Must complete in < 20 minutes
  - Runs on 2 vCPU / 8 GB RAM (no GPU)
"""

import os
import json
import httpx
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env if present
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — all from env vars as required
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)
MODEL = MODEL_NAME

# The OpenEnv server base URL — defaults to localhost for local testing
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = [
    "single_patient_easy",
    "concurrent_patients_medium",
    "incomplete_vitals_hard",
]

EPISODES_PER_TASK = 3  # Run 3 episodes per task to generate meaningful scores


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_reset(task_name: str, session_id: str) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict, session_id: str) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def env_state(session_id: str) -> dict:
    resp = httpx.get(
        f"{ENV_BASE_URL}/state",
        params={"session_id": session_id},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Triage Agent — with few-shot examples for better accuracy
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert emergency medicine physician trained in the Emergency Severity Index (ESI) triage system. You have 15 years of experience in Indian emergency departments.

## ESI Triage Levels

- **ESI 1 (Immediate/Critical)**: Life/limb/organ threat, requires immediate life-saving intervention. Patient is dying or has imminent threat to life.
  - Key indicators: No pulse, not breathing, severe respiratory distress with cyanosis, unresponsive, active seizure, massive hemorrhage, anaphylaxis with airway compromise.
  - Vitals: BP <70 systolic, SpO2 <85%, HR >150 or <40 (adults), RR >35 or apneic.

- **ESI 2 (Emergent/High Risk)**: High-risk situation OR severe pain/distress OR confused/lethargic/disoriented. Should not wait.
  - Key indicators: Chest pain with cardiac risk factors, stroke symptoms (FAST), sepsis signs (fever+confusion+tachycardia), overdose, suicidal, acute psychosis.
  - Vitals: BP <90 systolic, SpO2 88-93%, HR 120-150, temp >39.5°C with altered mental status.

- **ESI 3 (Urgent)**: Stable but needs multiple resources (labs, imaging, IV meds, procedures). Could deteriorate.
  - Key indicators: Abdominal pain needing workup, possible fracture, kidney stone, asthma exacerbation, moderate fever needing tests.
  - Vitals: Mostly normal with specific abnormalities (moderate fever, slightly elevated HR).

- **ESI 4 (Less Urgent)**: Needs one resource (sutures, X-ray, prescription). Stable vitals.
  - Key indicators: Simple laceration, ear infection, ankle sprain, UTI symptoms, minor burns.
  - Vitals: Normal or near-normal.

- **ESI 5 (Non-Urgent)**: No resources needed beyond physician exam. Can wait safely.
  - Key indicators: Prescription refill, cold symptoms, suture removal, routine concerns.
  - Vitals: Completely normal.

## Few-Shot Examples

**Example 1** — 72-year-old, HR 0, BP 0/0, SpO2 0%, chief complaint: "Found unresponsive, no pulse"
→ ESI 1. Cardiac arrest requiring immediate CPR and resuscitation. All vital signs absent.

**Example 2** — 58-year-old, HR 110, BP 185/110, SpO2 93%, chief complaint: "Crushing chest pain with sweating"
→ ESI 2. Acute coronary syndrome presentation. High-risk cardiac event requiring emergent evaluation.

**Example 3** — 40-year-old, HR 95, BP 130/85, SpO2 96%, chief complaint: "Kidney stone, severe flank pain"
→ ESI 3. Needs multiple resources (labs, imaging, IV pain management) but hemodynamically stable.

**Example 4** — 22-year-old, HR 75, BP 118/76, SpO2 99%, chief complaint: "Cut on finger, needs stitches"
→ ESI 4. Minor laceration needing one resource (sutures). Normal vitals, no acute danger.

**Example 5** — 28-year-old, HR 72, BP 115/74, SpO2 99%, chief complaint: "Prescription refill"
→ ESI 5. No resources needed. Completely stable. Routine request.

## Critical Rules
- Missing vitals (shown as 0) suggest the patient may be too unstable to measure — this is DANGEROUS, triage higher.
- Ambulance arrival suggests higher acuity than walk-in.
- Always consider the worst-case scenario for the chief complaint.
- Respond with a JSON object ONLY — no markdown, no explanation outside the JSON."""


def build_single_patient_prompt(observation: dict) -> str:
    return f"""Triage this emergency department patient and assign an ESI level (1-5).

Patient Data:
- Patient ID: {observation['patient_id']}
- Age: {observation['age']} years
- Heart Rate: {observation['heart_rate']} BPM
- Blood Pressure: {observation['blood_pressure_systolic']}/{observation['blood_pressure_diastolic']} mmHg
- Oxygen Saturation (SpO2): {observation['oxygen_saturation']}%
- Respiratory Rate: {observation['respiratory_rate']} breaths/min
- Temperature: {observation['temperature']}°C
- Pain Scale: {observation['pain_scale']}/10
- Chief Complaint: "{observation['chief_complaint']}"
- Arrival Mode: {observation['arrival_mode']}
- Vitals Complete: {observation['vitals_complete']}

Respond with ONLY this JSON format:
{{
  "triage_level": <integer 1-5>,
  "reasoning": "<detailed clinical reasoning referencing specific vital signs and ESI criteria>",
  "confidence": <float 0.0-1.0>
}}"""


def build_concurrent_patients_prompt(observations: list) -> str:
    patients_text = ""
    for i, obs in enumerate(observations, 1):
        patients_text += f"""
Patient {i} (ID: {obs['patient_id']}):
  - Age: {obs['age']} | HR: {obs['heart_rate']} BPM | BP: {obs['blood_pressure_systolic']}/{obs['blood_pressure_diastolic']} mmHg
  - SpO2: {obs['oxygen_saturation']}% | RR: {obs['respiratory_rate']} | Temp: {obs['temperature']}°C
  - Pain: {obs['pain_scale']}/10 | Arrival: {obs['arrival_mode']}
  - Complaint: "{obs['chief_complaint']}"
"""

    patient_ids = [obs['patient_id'] for obs in observations]

    return f"""Five patients have arrived simultaneously in the ER. Rank them by triage priority (most critical first) using ESI criteria.

{patients_text}

You must rank all five patients from most urgent (ESI 1) to least urgent (ESI 5).

Respond with ONLY this JSON format:
{{
  "triage_level": 1,
  "reasoning": {json.dumps(json.dumps(patient_ids))},
  "confidence": <float 0.0-1.0>
}}

CRITICAL: The "reasoning" field must be a valid JSON-encoded array string of patient IDs in YOUR priority order (most urgent first).
For example: {json.dumps(json.dumps(patient_ids))}

Think carefully about each patient's vital signs and chief complaint before ranking."""


def build_hard_task_prompt(observation: dict, step: int) -> str:
    vitals_note = ""
    if not observation.get("vitals_complete", True):
        vitals_note = "\n\n⚠️ IMPORTANT: Some vitals are missing (shown as 0). You have two choices:\n1. Set triage_level=0 to REQUEST COMPLETE VITALS (costs 1 step and -0.05 reward penalty)\n2. Make your best clinical judgment NOW based on available information and chief complaint"

    # Format vitals, marking missing ones clearly
    def fmt_vital(val, unit, missing_text="NOT MEASURED"):
        if val == 0 or val == 0.0:
            return missing_text
        return f"{val} {unit}"

    return f"""HARD TRIAGE — Incomplete Vitals (Step {step}/3)

Patient Data:
- Patient ID: {observation['patient_id']}
- Age: {observation['age']} years
- Heart Rate: {fmt_vital(observation['heart_rate'], 'BPM')}
- Blood Pressure: {fmt_vital(observation['blood_pressure_systolic'], f"/{observation['blood_pressure_diastolic']} mmHg") if observation['blood_pressure_systolic'] != 0 else 'NOT MEASURED'}
- Oxygen Saturation (SpO2): {fmt_vital(observation['oxygen_saturation'], '%')}
- Respiratory Rate: {fmt_vital(observation['respiratory_rate'], 'breaths/min')}
- Temperature: {fmt_vital(observation['temperature'], '°C')}
- Pain Scale: {observation['pain_scale']}/10
- Chief Complaint: "{observation['chief_complaint']}"
- Arrival Mode: {observation['arrival_mode']}
- Vitals Complete: {observation['vitals_complete']}{vitals_note}

Respond with ONLY this JSON format:
{{
  "triage_level": <0 to request vitals OR 1-5 for final triage>,
  "reasoning": "<clinical reasoning explaining your decision>",
  "confidence": <float 0.0-1.0>
}}"""


def call_llm(prompt: str, retries: int = 2) -> dict:
    """Call LLM and parse JSON response into action dict. Retries on parse failure."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.1,  # Low temperature for consistency
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code blocks if present
            if "```" in content:
                # Handle ```json ... ``` and ``` ... ```
                code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
                if code_match:
                    content = code_match.group(1).strip()

            action = json.loads(content)
            # Ensure required fields with valid defaults
            action.setdefault("triage_level", 3)
            action.setdefault("reasoning", "Clinical assessment based on presented vitals and chief complaint.")
            action.setdefault("confidence", 0.5)

            # Clamp values
            action["triage_level"] = max(0, min(5, int(action["triage_level"])))
            action["confidence"] = max(0.0, min(1.0, float(action["confidence"])))

            return action

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < retries:
                continue  # Retry
            # Final fallback
            return {
                "triage_level": 3,
                "reasoning": f"Unable to parse LLM response, defaulting to ESI-3. Error: {str(e)}",
                "confidence": 0.3,
            }
        except Exception as e:
            return {
                "triage_level": 3,
                "reasoning": f"LLM call failed: {str(e)}. Defaulting to ESI-3.",
                "confidence": 0.2,
            }


# ─────────────────────────────────────────────────────────────────────────────
#  Episode runners per task type
# ─────────────────────────────────────────────────────────────────────────────

def run_easy_episode(ep_num: int, session_id: str) -> tuple:
    """Returns (total_reward, steps)"""
    task_name = "single_patient_easy"
    observation = env_reset(task_name, session_id)
    step = 0
    total_reward = 0.0
    rewards: List[float] = []

    log_start(task=task_name, env="ClinicalTriage-Env", model=MODEL)

    done = False
    while not done:
        step += 1
        prompt = build_single_patient_prompt(observation)
        action = call_llm(prompt)

        result = env_step(action, session_id)
        reward = result["reward"]
        done = result["done"]
        total_reward += reward
        rewards.append(reward)

        action_json = json.dumps({"triage_level": action["triage_level"], "confidence": action["confidence"]})
        log_step(step=step, action=action_json, reward=reward, done=done, error=None)

        if not done:
            observation = result["observation"]

    success = total_reward > 0.0
    log_end(success=success, steps=step, score=total_reward, rewards=rewards)
    return total_reward, step


def run_medium_episode(ep_num: int, session_id: str) -> tuple:
    """Returns (total_reward, steps)"""
    task_name = "concurrent_patients_medium"
    first_obs = env_reset(task_name, session_id)

    # Fetch state to get all 5 patients
    state = env_state(session_id)
    all_observations = state.get("current_patients", [first_obs])

    step = 0
    total_reward = 0.0
    rewards: List[float] = []

    log_start(task=task_name, env="ClinicalTriage-Env", model=MODEL)

    step += 1
    prompt = build_concurrent_patients_prompt(all_observations)
    action = call_llm(prompt)

    result = env_step(action, session_id)
    reward = result["reward"]
    done = result["done"]
    total_reward += reward
    rewards.append(reward)

    action_json = json.dumps({"triage_level": action["triage_level"], "confidence": action["confidence"]})
    log_step(step=step, action=action_json, reward=reward, done=done, error=None)

    success = total_reward > 0.0
    log_end(success=success, steps=step, score=total_reward, rewards=rewards)
    return total_reward, step


def run_hard_episode(ep_num: int, session_id: str) -> tuple:
    """Returns (total_reward, steps)"""
    task_name = "incomplete_vitals_hard"
    observation = env_reset(task_name, session_id)
    step = 0
    total_reward = 0.0
    rewards: List[float] = []

    log_start(task=task_name, env="ClinicalTriage-Env", model=MODEL)

    done = False
    while not done:
        step += 1
        prompt = build_hard_task_prompt(observation, step)
        action = call_llm(prompt)

        result = env_step(action, session_id)
        reward = result["reward"]
        done = result["done"]
        total_reward += reward
        rewards.append(reward)

        action_json = json.dumps({"triage_level": action["triage_level"], "confidence": action["confidence"]})
        log_step(step=step, action=action_json, reward=reward, done=done, error=None)

        if not done and result.get("observation"):
            observation = result["observation"]

    success = total_reward > 0.0
    score = min(max(total_reward, 0.0), 1.0)
    log_end(success=success, steps=step, score=score, rewards=rewards)
    return total_reward, step


# ─────────────────────────────────────────────────────────────────────────────
#  Main — with summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            resp = httpx.get(f"{ENV_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"Server at {ENV_BASE_URL} did not respond after {max_retries} retries.")

    task_runners = {
        "single_patient_easy": run_easy_episode,
        "concurrent_patients_medium": run_medium_episode,
        "incomplete_vitals_hard": run_hard_episode,
    }

    # ── Run all episodes ─────────────────────────────────────────────────────
    all_results: dict = {t: [] for t in TASKS}

    for task_name in TASKS:
        runner = task_runners[task_name]
        for ep in range(1, EPISODES_PER_TASK + 1):
            session_id = f"{task_name}_ep{ep}_{int(time.time())}"
            try:
                reward, steps = runner(ep, session_id)
                all_results[task_name].append({
                    "episode": ep,
                    "reward": reward,
                    "steps": steps,
                })
            except Exception as exc:
                print(f"[ERROR] task={task_name} episode={ep} error={exc}")
                all_results[task_name].append({
                    "episode": ep,
                    "reward": 0.0,
                    "steps": 0,
                    "error": str(exc),
                })

    # ── Summary Statistics ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 65)
    print("  ClinicalTriage-Env — Inference Summary")
    print("=" * 65)
    print(f"  Model:          {MODEL}")
    print(f"  Total Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Episodes/Task:  {EPISODES_PER_TASK}")
    print("-" * 65)
    print(f"  {'Task':<35} {'Avg Reward':>10} {'Episodes':>10}")
    print("-" * 65)

    total_avg = 0.0
    for task_name in TASKS:
        results = all_results[task_name]
        rewards = [r["reward"] for r in results if "error" not in r]
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        total_avg += avg
        print(f"  {task_name:<35} {avg:>10.4f} {len(results):>10}")

    overall_avg = total_avg / len(TASKS)
    print("-" * 65)
    print(f"  {'OVERALL AVERAGE':<35} {overall_avg:>10.4f}")
    print("=" * 65)
    print(f"\n[INFO] Total inference time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
