# client.py
"""
ClinicalTriage-Env client SDK.

Async-first with .sync() wrapper for synchronous usage.

Async usage:
    async with ClinicalTriageEnvClient(base_url="ws://localhost:7860") as env:
        obs = await env.reset(task_name="single_patient_easy")
        result = await env.step(TriageAction(...))

Sync usage:
    with ClinicalTriageEnvClient(base_url="ws://localhost:7860").sync() as env:
        obs = env.reset()
        result = env.step(TriageAction(...))
"""
from __future__ import annotations
from openenv.core import EnvClient
from models import (
    PatientObservation, TriageAction, StepResult,
    EnvironmentState,
)


class ClinicalTriageEnvClient(EnvClient):

    async def reset(
        self,
        task_name: str = "single_patient_easy",
        set_index: int = 0,
    ) -> dict:
        """
        Reset the environment and return the first observation.

        Returns a dict with key 'observation' (easy/hard) or 'patients' (medium).
        """
        raw = await self._reset({"task_name": task_name, "set_index": set_index})
        if "patients" in raw:
            return {"patients": [PatientObservation(**p) for p in raw["patients"]]}
        return {"observation": PatientObservation(**raw["observation"])}

    async def step(self, action: TriageAction) -> StepResult:
        raw = await self._step(action.model_dump())
        return StepResult(**raw)

    async def state(self) -> EnvironmentState:
        raw = await self._state()
        return EnvironmentState(**raw)

    async def close(self):
        await self._close()

    async def run_episode(
        self,
        agent_fn,
        task_name: str = "single_patient_easy",
    ) -> dict:
        """
        Run a full episode with a callable agent.

        agent_fn: async callable (obs_dict) -> TriageAction
        Returns: dict with total_reward, steps, done, task_name
        """
        obs = await self.reset(task_name=task_name)
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = await agent_fn(obs)
            result = await self.step(action)
            total_reward += result.reward
            steps += 1
            done = result.done
            if result.observation:
                obs = {"observation": result.observation}
        return {
            "total_reward": total_reward,
            "steps": steps,
            "done": done,
            "task_name": task_name,
        }
