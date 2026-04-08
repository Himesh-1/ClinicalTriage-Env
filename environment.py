"""
environment.py — PatientSimulator and ESIGrader

PatientSimulator:
  - generate(task_name, profile_index) → PatientObservation
  - generate_all(task_name, set_index) → list for Medium task
  - reveal_vitals(raw_profile) → updated PatientObservation for Hard task
  - generate_procedural(esi_level) → novel patient from distributions
  - Seeded with numpy (seed=42) for reproducibility
  - Ground truth ESI is deterministically derived from vitals, not random

ESIGrader:
  - grade(action, ground_truth_esi, step_count) → float 0.0–1.0
  - Three-component reward: exact match + proximity + reasoning
"""

import os
import re
import json
import numpy as np
from typing import Optional, Tuple, List

from models import PatientObservation, TriageAction
from tasks import get_task
from dotenv import load_dotenv

# Load environment variables for local testing
load_dotenv()

# Seed for reproducibility as required by spec
_rng = np.random.default_rng(42)

# Counter for deterministic profile cycling across episodes
_profile_counter: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Procedural Patient Generator — ESI-level distributions
# ─────────────────────────────────────────────────────────────────────────────

_VITALS_DISTRIBUTIONS = {
    1: {  # Critical
        "heart_rate": (130, 180),
        "bp_systolic": (50, 85),
        "bp_diastolic": (25, 50),
        "oxygen_saturation": (70.0, 88.0),
        "respiratory_rate": (28, 45),
        "temperature": (35.0, 41.0),
        "pain_scale": (8, 10),
        "age_range": (1, 90),
    },
    2: {  # Emergent
        "heart_rate": (100, 130),
        "bp_systolic": (80, 100),
        "bp_diastolic": (50, 65),
        "oxygen_saturation": (88.0, 94.0),
        "respiratory_rate": (22, 30),
        "temperature": (36.0, 40.0),
        "pain_scale": (6, 9),
        "age_range": (5, 85),
    },
    3: {  # Urgent
        "heart_rate": (80, 105),
        "bp_systolic": (120, 145),
        "bp_diastolic": (78, 95),
        "oxygen_saturation": (94.0, 97.0),
        "respiratory_rate": (16, 22),
        "temperature": (37.0, 39.0),
        "pain_scale": (4, 8),
        "age_range": (10, 75),
    },
    4: {  # Less Urgent
        "heart_rate": (65, 85),
        "bp_systolic": (110, 130),
        "bp_diastolic": (70, 85),
        "oxygen_saturation": (97.0, 100.0),
        "respiratory_rate": (13, 17),
        "temperature": (37.0, 38.5),
        "pain_scale": (2, 5),
        "age_range": (5, 70),
    },
    5: {  # Non-Urgent
        "heart_rate": (60, 80),
        "bp_systolic": (110, 135),
        "bp_diastolic": (70, 85),
        "oxygen_saturation": (98.0, 100.0),
        "respiratory_rate": (12, 16),
        "temperature": (36.5, 37.2),
        "pain_scale": (0, 2),
        "age_range": (15, 80),
    },
}

_CHIEF_COMPLAINTS = {
    1: [
        "Cardiac arrest, CPR in progress",
        "Severe respiratory distress, cannot speak",
        "Unresponsive, no gag reflex",
        "Massive hemorrhage from trauma",
        "Active seizure for over 5 minutes",
        "Severe anaphylaxis, airway closing",
        "Gunshot wound to chest, profuse bleeding",
    ],
    2: [
        "Crushing chest pain with diaphoresis and nausea",
        "Sudden one-sided weakness, speech difficulty",
        "High fever with confusion and neck stiffness",
        "Intentional drug overdose, drowsy and confused",
        "Severe abdominal pain with rigid abdomen",
        "Acute psychosis, threatening self-harm",
        "Burns covering >20% body surface area",
    ],
    3: [
        "Moderate fever with productive cough for 5 days",
        "Kidney stone, severe colicky flank pain",
        "Possible arm fracture after fall, deformity visible",
        "Asthma exacerbation not responding to inhaler",
        "Abdominal pain with nausea, suspect appendicitis",
        "Vaginal bleeding in early pregnancy",
        "Severe migraine with vomiting, photophobia",
    ],
    4: [
        "Minor laceration needing stitches, bleeding controlled",
        "Ear infection, pain for 2 days, mild fever",
        "Ankle sprain, can bear weight, mild swelling",
        "Urinary symptoms, burning, frequency for 3 days",
        "Mild allergic rash after new medication",
        "Foreign body in eye, tearing, no vision changes",
        "Minor dog bite, superficial wound",
    ],
    5: [
        "Prescription refill needed, no acute complaints",
        "Mild cold symptoms, runny nose, no fever",
        "Suture removal from healed wound",
        "Wants tetanus booster, routine immunization",
        "Mild chronic knee pain, worsening over months",
        "Insomnia for 2 weeks, wants evaluation",
        "Requests medical certificate for school",
    ],
}


def generate_procedural(esi_level: int, patient_id: str) -> Tuple[PatientObservation, int]:
    """
    Generate a clinically plausible patient from ESI-level distributions.
    Uses seeded numpy RNG for reproducibility.
    """
    dist = _VITALS_DISTRIBUTIONS[esi_level]
    complaints = _CHIEF_COMPLAINTS[esi_level]

    age = int(_rng.integers(*dist["age_range"]))
    hr = int(_rng.integers(*dist["heart_rate"]))
    bp_s = int(_rng.integers(*dist["bp_systolic"]))
    bp_d = int(_rng.integers(*dist["bp_diastolic"]))
    spo2 = round(float(_rng.uniform(*dist["oxygen_saturation"])), 1)
    rr = int(_rng.integers(*dist["respiratory_rate"]))
    temp = round(float(_rng.uniform(*dist["temperature"])), 1)
    pain = int(_rng.integers(*dist["pain_scale"]))
    complaint = complaints[int(_rng.integers(0, len(complaints)))]
    arrival = "ambulance" if esi_level <= 2 else "walk-in"

    obs = PatientObservation(
        patient_id=patient_id,
        age=age,
        heart_rate=hr,
        blood_pressure_systolic=bp_s,
        blood_pressure_diastolic=bp_d,
        oxygen_saturation=spo2,
        respiratory_rate=rr,
        temperature=temp,
        pain_scale=pain,
        chief_complaint=complaint,
        arrival_mode=arrival,
        vitals_complete=True,
    )
    return obs, esi_level


# ─────────────────────────────────────────────────────────────────────────────
#  PatientSimulator
# ─────────────────────────────────────────────────────────────────────────────

class PatientSimulator:
    """
    Generates PatientObservation objects from task profiles.
    Supports both fixed-profile lookup and procedural generation.
    """

    @staticmethod
    def generate(task_name: str, profile_index: Optional[int] = None) -> Tuple[PatientObservation, int, dict]:
        """
        Returns:
          (PatientObservation, ground_truth_esi, raw_profile_dict)

        profile_index: if None, cycles deterministically through profiles.
        """
        task = get_task(task_name)
        profiles = task["patient_profiles"]

        if profile_index is None:
            # Deterministic cycling — different profile each call
            key = task_name
            if key not in _profile_counter:
                _profile_counter[key] = 0
            idx = _profile_counter[key] % len(profiles)
            _profile_counter[key] += 1
        else:
            idx = profile_index % len(profiles)

        raw = profiles[idx]

        # Build PatientObservation (strip fields not in model)
        obs_fields = {
            "patient_id": raw["patient_id"],
            "age": raw["age"],
            "heart_rate": raw["heart_rate"],
            "blood_pressure_systolic": raw["blood_pressure_systolic"],
            "blood_pressure_diastolic": raw["blood_pressure_diastolic"],
            "oxygen_saturation": raw["oxygen_saturation"],
            "respiratory_rate": raw["respiratory_rate"],
            "temperature": raw["temperature"],
            "pain_scale": raw["pain_scale"],
            "chief_complaint": raw["chief_complaint"],
            "arrival_mode": raw["arrival_mode"],
            "vitals_complete": raw["vitals_complete"],
        }
        observation = PatientObservation(**obs_fields)
        ground_truth_esi: int = raw["ground_truth_esi"]

        return observation, ground_truth_esi, raw

    @staticmethod
    def generate_all(task_name: str, set_index: int = 0) -> list:
        """
        For Medium task: return ALL patient profiles from a specific set as a
        list of (PatientObservation, ground_truth_esi, raw_profile) tuples.

        set_index cycles through available patient_sets.
        """
        task = get_task(task_name)

        # Use patient_sets if available (Medium task)
        if "patient_sets" in task and task["patient_sets"]:
            sets = task["patient_sets"]
            chosen = sets[set_index % len(sets)]
            profiles = chosen["profiles"]
        else:
            profiles = task["patient_profiles"]

        results = []
        for raw in profiles:
            obs_fields = {
                "patient_id": raw["patient_id"],
                "age": raw["age"],
                "heart_rate": raw["heart_rate"],
                "blood_pressure_systolic": raw["blood_pressure_systolic"],
                "blood_pressure_diastolic": raw["blood_pressure_diastolic"],
                "oxygen_saturation": raw["oxygen_saturation"],
                "respiratory_rate": raw["respiratory_rate"],
                "temperature": raw["temperature"],
                "pain_scale": raw["pain_scale"],
                "chief_complaint": raw["chief_complaint"],
                "arrival_mode": raw["arrival_mode"],
                "vitals_complete": raw["vitals_complete"],
            }
            obs = PatientObservation(**obs_fields)
            results.append((obs, raw["ground_truth_esi"], raw))
        return results

    @staticmethod
    def get_medium_ground_truth(task_name: str, set_index: int = 0) -> List[str]:
        """Return ground truth ranking for a specific patient set."""
        task = get_task(task_name)
        if "patient_sets" in task and task["patient_sets"]:
            sets = task["patient_sets"]
            chosen = sets[set_index % len(sets)]
            return chosen["ground_truth_ranking"]
        return task.get("ground_truth_ranking", [])

    @staticmethod
    def reveal_vitals(raw_profile: dict) -> PatientObservation:
        """
        For Hard task: merge the 'complete_vitals' patch into the profile
        and return an updated PatientObservation.
        """
        updated = dict(raw_profile)
        patch = updated.get("complete_vitals", {})
        updated.update(patch)

        obs_fields = {k: updated[k] for k in [
            "patient_id", "age", "heart_rate", "blood_pressure_systolic",
            "blood_pressure_diastolic", "oxygen_saturation", "respiratory_rate",
            "temperature", "pain_scale", "chief_complaint", "arrival_mode",
            "vitals_complete",
        ]}
        return PatientObservation(**obs_fields)


# ─────────────────────────────────────────────────────────────────────────────
#  ESIGrader — reward function
# ─────────────────────────────────────────────────────────────────────────────

def _extract_float_from_text(text: str) -> Optional[float]:
    """
    Robustly extract a float value from LLM output.
    Handles common patterns like "Score: 0.8", "0.75", "The rating is 0.9/1.0".
    """
    # Try direct float parse first
    text = text.strip()
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass

    # Regex: find patterns like 0.8, .75, 1.0
    matches = re.findall(r'(\d*\.?\d+)', text)
    for match in matches:
        try:
            val = float(match)
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            continue

    return None


def _llm_grade_reasoning(reasoning: str) -> float:
    """
    Grade the clinical reasoning using an LLM via the OpenAI client.
    Returns a float in [0.0, 1.0].

    Falls back to a heuristic score if env vars are missing (e.g. local dev).
    """
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")

    if not all([api_base, model_name, hf_token]):
        # Heuristic fallback: score by reasoning length and key clinical terms
        clinical_keywords = [
            "esi", "vital", "bp", "blood pressure", "heart rate", "oxygen",
            "spo2", "respiratory", "critical", "emergent", "urgent", "stable",
            "pain", "complaint", "triage", "priority", "immediate",
            "intervention", "life-threatening", "high-risk", "resource",
            "temperature", "fever", "saturation", "bradycardia", "tachycardia",
            "hypotension", "hypertension", "dyspnea", "altered mental",
        ]
        lower = reasoning.lower()
        keyword_hits = sum(1 for kw in clinical_keywords if kw in lower)
        length_score = min(len(reasoning) / 200, 1.0)
        keyword_score = min(keyword_hits / 5, 1.0)
        return round((length_score * 0.4 + keyword_score * 0.6), 4)

    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=hf_token)

        grading_prompt = f"""You are a senior emergency medicine physician grading a triage reasoning.
Score the following clinical reasoning on a scale of 0.0 to 1.0 based on:
- Correct identification of key clinical indicators
- Appropriate use of ESI triage criteria
- Logical and coherent reasoning
- Clinical accuracy

Reasoning to grade:
\"\"\"{reasoning}\"\"\"

Respond with ONLY a single float number between 0.0 and 1.0. No explanation."""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": grading_prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        score_text = response.choices[0].message.content.strip()
        score = _extract_float_from_text(score_text)
        if score is not None:
            return score
        return 0.5  # fallback
    except Exception:
        return 0.5


class ESIGrader:
    """
    Grades a TriageAction against the ground truth ESI level.

    Reward components:
      - Exact match:  0.6 if triage_level == ground_truth_esi, else 0.0
      - Proximity:    0.3 if |diff| == 1 (only when no exact match), else 0.0
      - Reasoning:    LLM score * 0.4
      - Step penalty (Hard task): -0.05 per step beyond step 1

    Total is clamped to [0.0, 1.0].
    """

    @staticmethod
    def grade(
        action: TriageAction,
        ground_truth_esi: int,
        step_count: int = 1,
    ) -> float:
        # ── Component 1: Exact match ──
        exact = 0.6 if action.triage_level == ground_truth_esi else 0.0

        # ── Component 2: Proximity (only when no exact match) ──
        if exact == 0.0:
            diff = abs(action.triage_level - ground_truth_esi)
            proximity = 0.3 if diff == 1 else 0.0
        else:
            proximity = 0.0

        # ── Component 3: LLM reasoning score ──
        reasoning_score = _llm_grade_reasoning(action.reasoning) * 0.4

        # ── Step penalty (Hard task: decay past step 1) ──
        step_penalty = max(0.0, (step_count - 1) * 0.05)

        # BUG FIX: Use addition instead of Python `or` on floats
        raw_reward = exact + proximity + reasoning_score - step_penalty
        return round(max(0.001, min(0.999, raw_reward)), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Kendall Tau rank correlation (for Medium task)
# ─────────────────────────────────────────────────────────────────────────────

def kendall_tau_reward(agent_ranking: list, ground_truth_ranking: list) -> float:
    """
    Compute normalised Kendall Tau correlation between two orderings.
    Returns a value in [0.0, 1.0] where 1.0 = perfect agreement.

    agent_ranking / ground_truth_ranking: lists of patient_id strings in order.
    """
    n = len(ground_truth_ranking)
    if n == 0:
        return 0.001

    # Map ground truth IDs to positions
    gt_pos = {pid: i for i, pid in enumerate(ground_truth_ranking)}

    # Build agent positions for IDs that appear in ground truth
    try:
        agent_pos = [gt_pos[pid] for pid in agent_ranking if pid in gt_pos]
    except KeyError:
        return 0.001

    if len(agent_pos) != n:
        return 0.001

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if agent_pos[i] < agent_pos[j]:
                concordant += 1
            elif agent_pos[i] > agent_pos[j]:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.999

    tau = (concordant - discordant) / total_pairs
    # Normalise from [-1, 1] → [0, 1]
    raw_reward = (tau + 1) / 2
    return round(max(0.001, min(0.999, raw_reward)), 4)
