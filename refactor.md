# ClinicalTriage-Env — Final Refactor Instructions

## Complete Implementation Guide for OpenEnv Parity + Hackathon Compliance

---

## Preamble — Read Before Touching Any File

This document is the **single source of truth** for the ClinicalTriage-Env refactor. It supersedes all prior planning documents.

### What must never change

- The ESI triage domain, 5-level scale, and clinical meaning
- The three-task progression: `single_patient_easy`, `concurrent_patients_medium`, `incomplete_vitals_hard`
- The LLM-as-judge reasoning grader (moved, not removed)
- The Kendall Tau ranking reward for the medium task (math stays identical)
- The numpy seed (42) for reproducibility
- The reward range: all rewards **must** be in `[0.0, 1.0]` — no 0.001/0.999 clamping

### What the hackathon validator actually checks

1. `POST /reset` with empty body `{}` returns HTTP 200
2. `openenv validate` passes on the repo directory
3. `docker build && docker run` succeeds cleanly
4. `inference.py` runs without error and emits correct stdout format
5. All task graders return scores in `[0.0, 1.0]`

### Hard constraints that override everything else

- `inference.py` **must** live at the root permanently — never move it
- `inference.py` **must** use `OpenAI` client for all LLM calls
- `inference.py` **must** emit only `[START]`, `[STEP]`, `[END]` to stdout — no other prints
- Total inference runtime **must** be under 20 minutes (3 tasks × 1 episode each)
- Must run on 2 vCPU / 8 GB RAM

---

## Final Target File Tree

This is the exact state the repository must be in when complete.

```
ClinicalTriage-Env/
├── .github/
│   └── workflows/
│       └── test.yml
├── eval/
│   ├── __init__.py
│   └── benchmark.py
├── examples/
│   ├── __init__.py
│   └── grpo_training.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── clinical_triage_environment.py
├── tasks/
│   ├── single_patient_easy.json
│   ├── concurrent_patients_medium.json
│   └── incomplete_vitals_hard.json
├── tests/
│   ├── test_compliance.py        (existing — update, do not delete)
│   ├── test_client.py            (new)
│   └── test_rubrics.py           (new)
├── __init__.py                   (new)
├── .env.example                  (update)
├── .gitignore                    (keep as-is)
├── client.py                     (new)
├── Dockerfile                    (update CMD only)
├── environment.py                (refactor — remove module-level mutable state)
├── inference.py                  (REWRITE — stays at root, never moves)
├── models.py                     (update)
├── openenv.yaml                  (update)
├── pyproject.toml                (update)
├── README.md                     (update)
├── rubrics.py                    (new — replaces grader.py)
├── runner.py                     (new)
├── tasks.py                      (rewrite as registry loader)
└── uv.lock
```

### Files to delete

```
grader.py
howto.txt
INSTRUCTIONS.md
```

### Files that must NOT be moved

```
inference.py   ← hackathon constraint, always at root
```

---

## Phase 1 — Task Config Extraction

### 1.1 Create `tasks/` directory

Extract all hardcoded patient profile data from `tasks.py` into three JSON files.

**`tasks/single_patient_easy.json`**

```json
{
  "name": "single_patient_easy",
  "difficulty": "easy",
  "description": "Single patient with unambiguous vitals. Reward based on exact ESI match plus reasoning quality.",
  "interaction_mode": "single_step",
  "max_steps": 1,
  "patient_profiles": [
    {
      "patient_id": "P001",
      "age": 65,
      "heart_rate": 130,
      "blood_pressure_systolic": 80,
      "blood_pressure_diastolic": 50,
      "oxygen_saturation": 88.0,
      "respiratory_rate": 28,
      "temperature": 37.2,
      "pain_scale": 9,
      "chief_complaint": "I can't breathe, severe chest pain",
      "arrival_mode": "ambulance",
      "vitals_complete": true,
      "ground_truth_esi": 1
    }
  ]
}
```

Create `tasks/concurrent_patients_medium.json` and `tasks/incomplete_vitals_hard.json` by extracting the existing hardcoded data from `tasks.py`. Do not invent new patient profiles — extract exactly what is already there.

`incomplete_vitals_hard.json` must include a `complete_vitals` patch object on each profile (the data already exists in the current `tasks.py` hard task profiles).

Set `"interaction_mode": "multi_patient"` for the medium task and `"interaction_mode": "incomplete_vitals"` for the hard task.

### 1.2 Rewrite `tasks.py` as a registry loader

```python
# tasks.py
import json
from pathlib import Path

_TASKS_DIR = Path(__file__).parent / "tasks"


def get_task(task_name: str) -> dict:
    path = _TASKS_DIR / f"{task_name}.json"
    if not path.exists():
        raise ValueError(
            f"Unknown task: {task_name!r}. Available: {list_tasks()}"
        )
    return json.loads(path.read_text())


def list_tasks() -> list[str]:
    return [p.stem for p in sorted(_TASKS_DIR.glob("*.json"))]
```

---

## Phase 2 — Environment Subclass

Create `server/__init__.py` (empty file).

Create `server/clinical_triage_environment.py`. This class owns **all** mutable episode state on `self`. There must be **zero module-level mutable variables** — this is what enables per-session isolation for parallel RL training workers.

```python
# server/clinical_triage_environment.py
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

    def __init__(self):
        self._task_name: str = "single_patient_easy"
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._current_patients: list[PatientObservation] = []
        self._ground_truth: list[int] = []
        self._raw_profiles: list[dict] = []
        self._set_index: int = 0
        self._rubric = ClinicalTriageRubric()

    async def reset(self, config: dict | None = None) -> dict:
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
            return {
                "patients": [p.model_dump() for p in self._current_patients],
                "task_name": task_name,
            }
        else:
            obs, esi, raw = PatientSimulator.generate(task_name)
            self._current_patients = [obs]
            self._ground_truth = [esi]
            self._raw_profiles = [raw]
            return {
                "observation": obs.model_dump(),
                "task_name": task_name,
            }

    async def step(self, action: dict) -> dict:
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
                return {
                    "observation": revealed_obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "action": "vitals_revealed",
                        "step": self._step_count,
                    },
                }
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

        return {
            "observation": None,
            "reward": reward,
            "done": self._done,
            "info": {
                "step": self._step_count,
                "cumulative_reward": self._cumulative_reward,
                "task_name": self._task_name,
            },
        }

    async def state(self) -> dict:
        return EnvironmentState(
            task_name=self._task_name,
            step_count=self._step_count,
            current_patients=self._current_patients,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
        ).model_dump()
```

### Refactor `environment.py`

Remove the module-level `_profile_counter` dict. Replace it with instance-level counters inside `PatientSimulator` or pass the index explicitly. All other logic (distributions, chief complaints, `generate_procedural`, `reveal_vitals`) stays exactly as-is.

---

## Phase 3 — Rubrics

Delete `grader.py`. Create `rubrics.py` at the project root.

```python
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
from models import PatientObservation, TriageAction


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

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
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
        concordant = discordant = 0
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
                raise ValueError
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
```

---

## Phase 4 — Server App

Create `server/app.py`. This replaces whatever currently acts as the server entry point.

```python
# server/app.py
"""
FastAPI entry point for ClinicalTriage-Env.
Uses openenv.core.HTTPEnvServer for WebSocket session management,
/ws endpoint, SIMULATION/PRODUCTION modes, and per-session isolation.
"""
from fastapi import FastAPI
from openenv.core import HTTPEnvServer, ServerMode
from .clinical_triage_environment import ClinicalTriageEnvironment
from tasks import list_tasks, get_task


def create_app(mode: ServerMode = ServerMode.SIMULATION) -> FastAPI:
    env_server = HTTPEnvServer(
        environment_class=ClinicalTriageEnvironment,
        mode=mode,
        max_concurrent_envs=64,
    )
    app = env_server.get_app()

    @app.get("/tasks")
    async def get_tasks():
        return [
            {
                "name": name,
                "difficulty": get_task(name).get("difficulty", "unknown"),
                "description": get_task(name).get("description", ""),
            }
            for name in list_tasks()
        ]

    return app


app = create_app()
```

Update `Dockerfile` CMD:

```dockerfile
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Phase 5 — Client SDK

Create `client.py` at the project root.

```python
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

    async def close(self) -> None:
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
```

Create `__init__.py` at the project root:

```python
# __init__.py
"""ClinicalTriage-Env — OpenEnv-compatible ESI patient triage benchmark."""
from .models import TriageAction as Action, PatientObservation as Observation
from .models import EnvironmentState as State
from .client import ClinicalTriageEnvClient as Env

__all__ = ["Action", "Observation", "State", "Env"]
```

---

## Phase 6 — Local Runner

Create `runner.py` at the project root.

```python
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

    async def reset(self, task_name: str = "single_patient_easy", **kwargs) -> dict:
        return await self._env.reset({"task_name": task_name, **kwargs})

    async def step(self, action: TriageAction) -> dict:
        return await self._env.step(action.model_dump())

    async def state(self) -> dict:
        return await self._env.state()


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
        steps += 1
        done = result.get("done", True)
    return {
        "total_reward": total_reward,
        "steps": steps,
        "done": done,
        "task_name": task_name,
    }
```

---

## Phase 7 — Evaluation Harness

Create `eval/__init__.py` (empty).

Create `eval/benchmark.py`.

**Critical:** This file must never import or call `inference.py` in any way. It is a separate tool for development iteration only.

```python
#!/usr/bin/env python3
# eval/benchmark.py
"""
ClinicalTriage-Env evaluation harness.

Development tool — completely separate from inference.py.
Never used by the hackathon validator.

Usage:
    python eval/benchmark.py --local --episodes 5
    python eval/benchmark.py --base-url ws://localhost:7860 --episodes 5
    python eval/benchmark.py --local --task single_patient_easy --episodes 3
"""
import argparse
import asyncio
import json
import os
import statistics
from typing import Optional

from models import TriageAction
from tasks import list_tasks


def _build_agent(api_base: str, model: str, token: str):
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=token)

    async def agent(obs: dict) -> TriageAction:
        prompt = (
            "You are a trained emergency triage nurse using the ESI scale.\n"
            f"Patient data: {json.dumps(obs, indent=2)}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"triage_level": <int 1-5>, "reasoning": "<clinical justification>", '
            '"confidence": <float 0-1>}'
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        return TriageAction(**data)

    return agent


async def run_benchmark(
    tasks: list[str],
    n_episodes: int,
    local: bool,
    base_url: str,
    api_base: str,
    model: str,
    token: str,
):
    agent = _build_agent(api_base, model, token)
    all_rewards = []

    for task_name in tasks:
        task_rewards = []
        print(f"\n{'─' * 52}")
        print(f"Task: {task_name}  ({n_episodes} episodes)")
        print(f"{'─' * 52}")

        for ep in range(n_episodes):
            if local:
                from runner import LocalTriageRunner, run_episode
                result = await run_episode(agent, task_name=task_name)
            else:
                from client import ClinicalTriageEnvClient
                async with ClinicalTriageEnvClient(base_url=base_url) as env:
                    result = await env.run_episode(agent, task_name=task_name)

            r = result["total_reward"]
            task_rewards.append(r)
            all_rewards.append(r)
            print(f"  Episode {ep + 1:02d}: reward={r:.4f}  steps={result['steps']}")

        mean = statistics.mean(task_rewards)
        std = statistics.stdev(task_rewards) if len(task_rewards) > 1 else 0.0
        print(f"  → mean={mean:.4f}  std={std:.4f}")

    print(f"\n{'═' * 52}")
    print(f"OVERALL AVERAGE: {statistics.mean(all_rewards):.4f}")
    print(f"{'═' * 52}\n")


def main():
    parser = argparse.ArgumentParser(description="ClinicalTriage-Env benchmark")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--base-url", default="ws://localhost:7860")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    token = os.environ.get("HF_TOKEN", "")
    tasks = [args.task] if args.task else list_tasks()

    asyncio.run(run_benchmark(
        tasks=tasks,
        n_episodes=args.episodes,
        local=args.local,
        base_url=args.base_url,
        api_base=api_base,
        model=model,
        token=token,
    ))


if __name__ == "__main__":
    main()
```

---

## Phase 8 — inference.py Rewrite (Hackathon Submission Script)

**This is the most critical file.** Rewrite `inference.py` in place at the root. Do not move it.

Rules that must hold:

- Uses `OpenAI` client for all LLM calls — no other HTTP library for LLM calls
- Reads `API_BASE_URL` (with default), `MODEL_NAME` (with default), `HF_TOKEN` (no default, raises if missing), `ENV_BASE_URL` (with default), `LOCAL_IMAGE_NAME` (with default)
- Emits **only** `[START]`, `[STEP]`, `[END]` to stdout — every other print must go to stderr or be removed
- `[END]` is always emitted — the `finally` block is non-negotiable
- `[END]` includes `score=` field formatted to 2 decimal places
- Runs exactly **one episode per task** — 3 tasks total — to stay within 20-minute limit
- Uses `requests` for env HTTP calls (no dependency on `client.py` or `openenv` SDK)
- Score per task = `sum(rewards) / len(rewards)` clamped to `[0.0, 1.0]`

````python
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
from typing import Optional

import requests
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME: str = os.getenv(
    "MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
)
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")
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
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
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

def _reset(task_name: str) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _step(action: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
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

    try:
        reset_data = _reset(task_name)

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
                result = _step(action_dict)
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
````

---

## Phase 9 — Models Update

Update `models.py`. Key changes only — do not rewrite models that are already correct.

```python
# models.py — updated sections

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class PatientObservation(BaseModel):
    patient_id: str
    age: int
    heart_rate: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    oxygen_saturation: float
    respiratory_rate: int
    temperature: float
    pain_scale: int
    chief_complaint: str
    arrival_mode: str
    vitals_complete: bool


class TriageAction(BaseModel):
    triage_level: int = Field(..., ge=0, le=5)
    reasoning: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class StepResult(BaseModel):
    # observation is Optional[PatientObservation] only — never a list.
    # Medium task patient list belongs in reset response under 'patients' key only.
    observation: Optional[PatientObservation] = None
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: Optional[PatientObservation] = None
    patients: Optional[List[PatientObservation]] = None
    task_name: str
    session_id: Optional[str] = None


class EnvironmentState(BaseModel):
    task_name: str
    step_count: int
    current_patients: List[PatientObservation]
    done: bool
    cumulative_reward: float


class ResetRequest(BaseModel):
    task_name: Optional[str] = "single_patient_easy"
    set_index: Optional[int] = 0
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: TriageAction
    session_id: Optional[str] = None


class TaskConfig(BaseModel):
    name: str
    difficulty: str
    description: str
    interaction_mode: str
    max_steps: int = 1
    patient_profiles: Optional[List[Dict[str, Any]]] = None
    patient_sets: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str


class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str
```

**Critical note on `ResetRequest`:** The validator pings `/reset` with an empty body `{}`. The `task_name` field must have a default value so empty-body requests succeed without validation errors.

---

## Phase 10 — Tests

### `tests/test_rubrics.py`

```python
# tests/test_rubrics.py
"""Unit tests for rubric components. No server, no LLM, no network needed."""
import pytest
from models import TriageAction, PatientObservation
from rubrics import (
    ExactMatchRubric, ProximityRubric, ReasoningRubric,
    KendallTauRubric, ClinicalTriageRubric,
)


def test_exact_match_hit():
    assert ExactMatchRubric().grade(2, 2) == 0.6


def test_exact_match_miss():
    assert ExactMatchRubric().grade(3, 1) == 0.0


def test_proximity_off_by_one():
    assert ProximityRubric().grade(2, 3, exact_matched=False) == 0.3


def test_proximity_off_by_two():
    assert ProximityRubric().grade(1, 3, exact_matched=False) == 0.0


def test_proximity_suppressed_on_exact():
    assert ProximityRubric().grade(2, 2, exact_matched=True) == 0.0


def test_reasoning_heuristic_in_range():
    r = ReasoningRubric()
    score = r.grade("ESI-1: HR 130, BP 80/50, SpO2 88% — critical vitals.", skip_llm=True)
    assert 0.0 <= score <= r.weight


def test_reasoning_empty_string():
    r = ReasoningRubric()
    score = r.grade("", skip_llm=True)
    assert score == 0.0


def test_kendall_tau_perfect():
    r = KendallTauRubric()
    gt = ["P001", "P002", "P003"]
    assert r.grade(gt, gt) == 1.0


def test_kendall_tau_reversed():
    r = KendallTauRubric()
    gt = ["P001", "P002", "P003"]
    assert r.grade(["P003", "P002", "P001"], gt) == 0.0


def test_kendall_tau_empty():
    assert KendallTauRubric().grade([], []) == 0.0


def test_reward_in_unit_interval():
    rubric = ClinicalTriageRubric()
    action = TriageAction(
        triage_level=1,
        reasoning="Critical: BP 80/50, SpO2 88%, HR 130. ESI-1.",
        confidence=0.95,
    )
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=1, skip_llm=True)
    assert 0.0 <= reward <= 1.0


def test_reward_never_exceeds_one():
    rubric = ClinicalTriageRubric()
    action = TriageAction(
        triage_level=1,
        reasoning="x" * 500,  # very long reasoning
        confidence=1.0,
    )
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=1, skip_llm=True)
    assert reward <= 1.0


def test_reward_never_below_zero():
    rubric = ClinicalTriageRubric()
    action = TriageAction(triage_level=5, reasoning="", confidence=0.0)
    reward = rubric.grade_easy(action, ground_truth_esi=1, step_count=10, skip_llm=True)
    assert reward >= 0.0
```

### `tests/test_client.py`

```python
# tests/test_client.py
"""Integration tests using LocalTriageRunner. No server process needed."""
import asyncio
import pytest
from runner import LocalTriageRunner
from models import TriageAction


@pytest.mark.asyncio
async def test_reset_easy_returns_observation():
    runner = LocalTriageRunner()
    obs = await runner.reset(task_name="single_patient_easy")
    assert "observation" in obs
    assert "patient_id" in obs["observation"]


@pytest.mark.asyncio
async def test_reset_medium_returns_patients():
    runner = LocalTriageRunner()
    obs = await runner.reset(task_name="concurrent_patients_medium")
    assert "patients" in obs
    assert len(obs["patients"]) == 5


@pytest.mark.asyncio
async def test_step_reward_in_range():
    runner = LocalTriageRunner()
    await runner.reset(task_name="single_patient_easy")
    action = TriageAction(
        triage_level=1,
        reasoning="Critical: low SpO2, low BP, high HR.",
        confidence=0.9,
    )
    result = await runner.step(action)
    assert "reward" in result
    assert 0.0 <= result["reward"] <= 1.0


@pytest.mark.asyncio
async def test_step_done_after_easy_task():
    runner = LocalTriageRunner()
    await runner.reset(task_name="single_patient_easy")
    action = TriageAction(triage_level=2, reasoning="Emergent patient.", confidence=0.8)
    result = await runner.step(action)
    assert result["done"] is True


@pytest.mark.asyncio
async def test_state_is_nondestructive():
    runner = LocalTriageRunner()
    await runner.reset(task_name="single_patient_easy")
    s1 = await runner.state()
    s2 = await runner.state()
    assert s1["step_count"] == s2["step_count"]


@pytest.mark.asyncio
async def test_reset_clears_episode():
    runner = LocalTriageRunner()
    await runner.reset(task_name="single_patient_easy")
    action = TriageAction(triage_level=3, reasoning="Urgent.", confidence=0.7)
    await runner.step(action)
    await runner.reset(task_name="single_patient_easy")
    s = await runner.state()
    assert s["step_count"] == 0
    assert s["done"] is False
```

---

## Phase 11 — Configuration Files

### `pyproject.toml`

```toml
[project]
name = "clinicaltriage-env"
version = "0.1.0"
description = "OpenEnv-compatible ESI patient triage RL benchmark"
requires-python = ">=3.11"
dependencies = [
    "openenv-core>=0.1.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.0.0",
    "numpy>=1.26.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "websockets>=12.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
eval = ["pandas>=2.0.0"]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "httpx>=0.27.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[project.scripts]
clinicaltriage-benchmark = "eval.benchmark:main"
```

### `openenv.yaml`

```yaml
name: clinicaltriage-env
version: "0.1.0"
description: "ESI patient triage RL benchmark — 3 task levels, LLM-graded reasoning rewards"
emoji: "🏥"
tags:
  - healthcare
  - triage
  - reasoning
  - llm-judge
  - rl-benchmark
interaction_mode: multi_step
client_class: ClinicalTriageEnvClient
action_class: TriageAction
observation_class: PatientObservation
tasks:
  - single_patient_easy
  - concurrent_patients_medium
  - incomplete_vitals_hard
hf_space: true
app_port: 7860
sdk: docker
colorFrom: blue
colorTo: green
pinned: false
```

### `.env.example`

```bash
# LLM API endpoint — required, has default
API_BASE_URL=https://router.huggingface.co/v1

# Model identifier — required, has default
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# HuggingFace / API key — required, no default, inference.py raises if missing
HF_TOKEN=hf_your_token_here

# Running environment server URL — used by inference.py and benchmark.py
ENV_BASE_URL=http://localhost:7860

# Docker image name for from_docker_image() pattern
LOCAL_IMAGE_NAME=clinicaltriage-env:latest

# Set to "true" to skip LLM-as-judge reasoning grader (speeds up local testing)
SKIP_LLM_GRADER=false
```

### `.github/workflows/test.yml`

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Rubric unit tests (no server, no LLM)
        run: uv run pytest tests/test_rubrics.py -v

      - name: Client integration tests (no server)
        run: uv run pytest tests/test_client.py -v
        env:
          SKIP_LLM_GRADER: "true"
```

---

## Phase 12 — README Updates

Add these sections to `README.md`. Do not rewrite existing content.

```markdown
## Installation

pip install clinicaltriage-env

## Quick start — client SDK

from clinicaltriage_env import Env, Action
import asyncio

async def main():
async with Env(base_url="ws://localhost:7860") as env:
obs = await env.reset(task_name="single_patient_easy")
action = Action(
triage_level=1,
reasoning="Critical: SpO2 88%, BP 80/50, HR 130",
confidence=0.95,
)
result = await env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(main())

## Evaluation harness

# Run 5 episodes per task locally (no server needed):

python eval/benchmark.py --local --episodes 5

# Run against a deployed server:

python eval/benchmark.py --base-url ws://localhost:7860 --episodes 5

## Environment variables

| Variable         | Required | Default                          | Description                    |
| ---------------- | -------- | -------------------------------- | ------------------------------ |
| API_BASE_URL     | Yes      | https://router.huggingface.co/v1 | LLM API endpoint               |
| MODEL_NAME       | Yes      | meta-llama/Llama-3.1-8B-Instruct | Model identifier               |
| HF_TOKEN         | Yes      | none                             | HuggingFace / API key          |
| ENV_BASE_URL     | No       | http://localhost:7860            | Running environment server URL |
| LOCAL_IMAGE_NAME | No       | clinicaltriage-env:latest        | Docker image name              |
| SKIP_LLM_GRADER  | No       | false                            | Skip LLM judge (local testing) |
```

---

## Acceptance Criteria

Run every item in this checklist before submitting. Each must pass.

### Code correctness

- [ ] `python -c "from clinicaltriage_env import Env, Action, Observation"` exits 0
- [ ] `uv run pytest tests/test_rubrics.py -v` — all pass, no server needed
- [ ] `uv run pytest tests/test_client.py -v SKIP_LLM_GRADER=true` — all pass, no server needed
- [ ] No reward anywhere in the codebase is clamped to 0.001 or 0.999
- [ ] No module-level mutable variables in `environment.py` or `server/clinical_triage_environment.py`

### Server

- [ ] `uvicorn server.app:app --port 7860` starts without error
- [ ] `curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'` returns HTTP 200
- [ ] `curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_name":"single_patient_easy"}'` returns a `PatientObservation`
- [ ] `curl -s http://localhost:7860/tasks` returns all 3 tasks
- [ ] `curl -s http://localhost:7860/health` returns `{"status": "ok"}`

### inference.py

- [ ] `inference.py` is at the root — never inside a subdirectory
- [ ] `python inference.py` stdout contains only lines starting with `[START]`, `[STEP]`, or `[END]`
- [ ] `[END]` line includes `score=` field
- [ ] `[END]` is printed even when an exception occurs mid-episode
- [ ] Script completes all 3 tasks in under 20 minutes
- [ ] `HF_TOKEN` unset → script raises `ValueError` immediately before any prints

### Docker

- [ ] `docker build -t clinicaltriage-env .` succeeds
- [ ] `docker run -p 7860:7860 clinicaltriage-env` boots and serves
- [ ] Container responds to `/reset` with empty body within 30 seconds of start

### OpenEnv spec

- [ ] `openenv validate .` passes
- [ ] `openenv.yaml` contains correct `name`, `tasks`, `app_port`, `sdk: docker`

### Hackathon pre-validation script

- [ ] `./validate-submission.sh https://your-space.hf.space` — all 3/3 checks pass

---

## What Not to Change

| Item                                             | Reason                                   |
| ------------------------------------------------ | ---------------------------------------- |
| `inference.py` location (root)                   | Hackathon disqualification if moved      |
| ESI 5-level clinical scale                       | Core domain                              |
| Three task names exactly as-is                   | Validator enumerates them by name        |
| Kendall Tau math in `KendallTauRubric`           | Reward correctness                       |
| Numpy seed 42 in `environment.py`                | Reproducibility requirement              |
| `ResetRequest.task_name` having a default        | Validator pings `/reset` with empty body |
| OpenAI client in `inference.py`                  | Hackathon requirement                    |
| `[START]`/`[STEP]`/`[END]` field names and order | Validator parses exact format            |
