# rubrics.py
"""
ClinicalTriage reward rubrics.

Composable rubric classes. ClinicalTriageRubric delegates to sub-rubrics.
All outputs clamped to [0.0, 1.0]. No 0.001/0.999 sentinel values.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional
from models import TriageAction, PatientObservation


class ExactMatchRubric:
    weight: float = 0.6

    def grade(self, predicted: int, ground_truth: int) -> float:
        return self.weight if predicted == ground_truth else 0.0


class ProximityRubric:
    weight: float = 0.3

    def grade(self, predicted: int, ground_truth: int, exact_matched: bool) -> float:
        if exact_matched:
            return 0.0
        return self.weight if abs(predicted - ground_truth) == 1 else 0.0


class ReasoningRubric:
    weight: float = 0.4

    _KEYWORDS = [
        "esi", "vital", "bp", "blood pressure", "heart rate", "oxygen",
        "spo2", "respiratory", "critical", "emergent", "urgent", "stable",
        "pain", "complaint", "triage", "priority", "immediate",
        "intervention", "life-threatening", "high-risk", "resource",
        "temperature", "fever", "saturation", "bradycardia", "tachycardia",
        "hypotension", "hypertension", "dyspnea", "altered mental",
    ]

    def grade(self, reasoning: str, skip_llm: bool = False) -> float:
        """Returns weighted score in [0.0, weight]."""
        raw = self._heuristic(reasoning) if skip_llm else self._llm_or_fallback(reasoning)
        return round(raw * self.weight, 4)

    def _llm_or_fallback(self, reasoning: str) -> float:
        api_base = os.environ.get("API_BASE_URL")
        model = os.environ.get("MODEL_NAME")
        token = os.environ.get("HF_TOKEN")
        if not all([api_base, model, token]):
            return self._heuristic(reasoning)
        try:
            from openai import OpenAI
            client = OpenAI(base_url=api_base, api_key=token)
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are a senior emergency physician grading triage reasoning.\n"
                        "Score 0.0–1.0 on: correct clinical indicators, ESI criteria, "
                        "logical coherence, clinical accuracy.\n\n"
                        f'Reasoning: """{reasoning}"""\n\n'
                        "Respond with ONLY a single float between 0.0 and 1.0."
                    ),
                }],
                max_tokens=10,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            return self._parse_float(text) or 0.5
        except Exception:
            return 0.5

    def _heuristic(self, reasoning: str) -> float:
        lower = reasoning.lower()
        hits = sum(1 for kw in self._KEYWORDS if kw in lower)
        length_score = min(len(reasoning) / 200, 1.0)
        keyword_score = min(hits / 5, 1.0)
        return round(length_score * 0.4 + keyword_score * 0.6, 4)

    def _parse_float(self, text: str) -> Optional[float]:
        try:
            v = float(text)
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
        for m in re.findall(r'(\d*\.?\d+)', text):
            try:
                v = float(m)
                if 0.0 <= v <= 1.0:
                    return v
            except ValueError:
                continue
        return None


class KendallTauRubric:

    def grade(self, agent_ranking: list[str], ground_truth_ranking: list[str]) -> float:
        n = len(ground_truth_ranking)
        if n == 0:
            return 0.0

        gt_pos = {pid: i for i, pid in enumerate(ground_truth_ranking)}
        try:
            agent_pos = [gt_pos[pid] for pid in agent_ranking if pid in gt_pos]
        except KeyError:
            return 0.0

        if len(agent_pos) != n:
            return 0.0

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
            return 1.0

        tau = (concordant - discordant) / total_pairs
        return round(max(0.0, min(1.0, (tau + 1) / 2)), 4)


class ClinicalTriageRubric:

    def __init__(self):
        self._exact = ExactMatchRubric()
        self._proximity = ProximityRubric()
        self._reasoning = ReasoningRubric()
        self._ranking = KendallTauRubric()

    def grade_easy(
        self,
        action: TriageAction,
        ground_truth_esi: int,
        step_count: int = 1,
        skip_llm: bool = False,
    ) -> float:
        exact = self._exact.grade(action.triage_level, ground_truth_esi)
        prox = self._proximity.grade(action.triage_level, ground_truth_esi, exact > 0)
        reason = self._reasoning.grade(action.reasoning, skip_llm=skip_llm)
        return round(max(0.0, min(1.0, exact + prox + reason)), 4)

    def grade_medium(
        self,
        action: TriageAction,
        ground_truth_esis: list[int],
        patients: list[PatientObservation],
        step_count: int = 1,
    ) -> float:
        try:
            agent_ranking = json.loads(action.reasoning)
            if not isinstance(agent_ranking, list):
                return 0.0
        except (json.JSONDecodeError, ValueError):
            return 0.0
        gt_ranking = [
            p.patient_id
            for p in sorted(
                patients,
                key=lambda p: ground_truth_esis[patients.index(p)],
            )
        ]
        return round(max(0.0, min(1.0, self._ranking.grade(agent_ranking, gt_ranking))), 4)

    def grade_hard(
        self,
        action: TriageAction,
        ground_truth_esi: int,
        raw_profile: dict,
        step_count: int,
        skip_llm: bool = False,
    ) -> tuple[float, Optional[PatientObservation]]:
        from environment import PatientSimulator
        if action.triage_level == 0:
            revealed = PatientSimulator.reveal_vitals(raw_profile)
            return 0.0, revealed
        exact = self._exact.grade(action.triage_level, ground_truth_esi)
        prox = self._proximity.grade(action.triage_level, ground_truth_esi, exact > 0)
        reason = self._reasoning.grade(action.reasoning, skip_llm=skip_llm)
        step_penalty = max(0.0, (step_count - 1) * 0.05)
        raw = exact + prox + reason - step_penalty
        return round(max(0.0, min(1.0, raw)), 4), None
