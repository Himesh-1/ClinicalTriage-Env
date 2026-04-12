"""
ClinicalTriageEnvironment — OpenEnv Environment subclass.

One instance = one isolated episode. HTTPEnvServer creates a fresh
instance per WebSocket session. No shared mutable state at module level.
"""
from __future__ import annotations
import os
from typing import Optional
from openenv.core import Environment
from models import PatientObservation, TriageAction, EnvironmentState
from tasks import get_task, list_tasks
from environment import PatientSimulator
from rubrics import ClinicalTriageRubric


class ClinicalTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._task_name: str = "single_patient_easy"
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._current_patients: list[PatientObservation] = []
        self._ground_truth: list[int] = []
        self._set_index: int = 0
        self._raw_profiles: list[dict] = []
        self._rubric = ClinicalTriageRubric()

    def reset(self, config: dict | None = None) -> PatientObservation | list[PatientObservation]:
        cfg = config or {}
        task_name = cfg.get("task_name", "single_patient_easy")
        self._set_index = cfg.get("set_index", 0)
        task = get_task(task_name)

        self._task_name = task_name
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0

        mode = task.get("interaction_mode", "single_step")

        if mode == "multi_patient":
            results = PatientSimulator.generate_all(task_name, self._set_index)
            self._current_patients = [r[0] for r in results]
            self._ground_truth = [r[1] for r in results]
            self._raw_profiles = [r[2] for r in results]
            # Add reward and done fields to each patient
            return [p.model_copy(update={"reward": None, "done": False}) for p in self._current_patients]
        else:
            obs, esi, raw = PatientSimulator.generate(task_name)
            self._current_patients = [obs]
            self._ground_truth = [esi]
            self._raw_profiles = [raw]
            # Add reward and done fields
            return obs.model_copy(update={"reward": None, "done": False})

    def step(self, action: dict) -> PatientObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        parsed = TriageAction(**action)
        task = get_task(self._task_name)
        mode = task.get("interaction_mode", "single_step")
        skip_llm = os.getenv("SKIP_LLM_GRADER", "false").lower() == "true"

        if mode == "multi_patient":
            reward = self._rubric.grade_medium(
                parsed,
                self._ground_truth,
                self._current_patients,
                self._step_count,
            )
        elif mode == "incomplete_vitals":
            reward, revealed_obs = self._rubric.grade_hard(
                parsed,
                self._ground_truth[0],
                self._raw_profiles[0],
                self._step_count,
                skip_llm=skip_llm,
            )
            if revealed_obs is not None:
                self._current_patients[0] = revealed_obs
                revealed_with_rewards = revealed_obs.model_copy(update={"reward": 0.0, "done": False})
                return revealed_with_rewards
        else:
            reward = self._rubric.grade_easy(
                parsed,
                self._ground_truth[0],
                self._step_count,
                skip_llm=skip_llm,
            )

        self._cumulative_reward += reward
        max_steps = task.get("max_steps", 1)
        self._done = (
            self._step_count >= max_steps or parsed.triage_level != 0
        )

        # Return the observation with reward and done attached
        obs_with_rewards = self._current_patients[0].model_copy(
            update={"reward": reward, "done": self._done}
        )
        return obs_with_rewards

    def state(self) -> dict:
        return EnvironmentState(
            task_name=self._task_name,
            step_count=self._step_count,
            current_patients=self._current_patients,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
        ).model_dump()
