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
