# runner.py
"""
Local runner — runs episodes without a network connection.
Directly instantiates ClinicalTriageEnvironment.
Useful for unit tests, offline eval, and training on the same machine.
"""
from __future__ import annotations
import asyncio
from typing import Callable, Awaitable
from models import TriageAction
from server.clinical_triage_environment import ClinicalTriageEnvironment


class LocalTriageRunner:
    """Same interface as ClinicalTriageEnvClient but requires no server."""

    def __init__(self):
        self._env = ClinicalTriageEnvironment()

    async def reset(self, task_name: str = "single_patient_easy") -> dict:
        obs = self._env.reset({"task_name": task_name})
        # Return in dict format for compatibility with agent functions
        if isinstance(obs, list):
            return {"patients": [p.model_dump() for p in obs]}
        else:
            return {"observation": obs.model_dump()}

    async def step(self, action: TriageAction) -> dict:
        obs = self._env.step(action.model_dump())
        # obs now includes reward and done fields
        return {
            "observation": obs.model_dump() if not isinstance(obs, dict) else obs,
            "reward": obs.reward if hasattr(obs, "reward") else 0.0,
            "done": obs.done if hasattr(obs, "done") else False,
        }

    async def state(self) -> dict:
        return self._env.state()


async def run_episode(
    agent_fn: Callable[[dict], Awaitable[TriageAction]],
    task_name: str = "single_patient_easy",
    runner: LocalTriageRunner | None = None,
) -> dict:
    r = runner or LocalTriageRunner()
    obs = await r.reset(task_name=task_name)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = await agent_fn(obs)
        result = await r.step(action)
        total_reward += result.get("reward", 0.0)
        if result.get("observation"):
            obs = {"observation": result["observation"]}
        steps += 1
        done = result.get("done", True)
    return {
        "total_reward": total_reward,
        "steps": steps,
        "done": done,
        "task_name": task_name,
    }
