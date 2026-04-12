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
        text = resp.choices[0].message.content.strip()
        try:
            parsed = json.loads(text)
            return TriageAction(**parsed)
        except Exception:
            return TriageAction(
                triage_level=3,
                reasoning="Fallback: LLM failed to parse.",
                confidence=0.1,
            )

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
                from runner import run_episode
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
    if all_rewards:
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
    tasks_to_run = [args.task] if args.task else list_tasks()

    asyncio.run(run_benchmark(
        tasks=tasks_to_run,
        n_episodes=args.episodes,
        local=args.local,
        base_url=args.base_url,
        api_base=api_base,
        model=model,
        token=token,
    ))


if __name__ == "__main__":
    main()
