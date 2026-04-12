"""
environment.py — PatientSimulator

PatientSimulator:
  - generate(task_name, profile_index) → PatientObservation
  - generate_all(task_name, set_index) → list for Medium task
  - reveal_vitals(raw_profile) → updated PatientObservation for Hard task
  - generate_procedural(esi_level) → novel patient from distributions
  - Seeded with numpy (seed=42) for reproducibility
  - Ground truth ESI is deterministically derived from vitals, not random
"""

import numpy as np
from typing import Optional, Tuple, List

from models import PatientObservation
from tasks import get_task
from dotenv import load_dotenv

# Load environment variables for local testing
load_dotenv()

# Seed for reproducibility as required by spec
_rng = np.random.default_rng(42)

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

        profile_index: if None, cycles randomly.
        """
        task = get_task(task_name)
        profiles = task["patient_profiles"]

        if profile_index is None:
            idx = int(_rng.integers(0, len(profiles)))
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
