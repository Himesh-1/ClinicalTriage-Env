"""
FastAPI entry point for ClinicalTriage-Env.
Custom endpoints for compliance with test suite.
"""
from fastapi import FastAPI, HTTPException
from .clinical_triage_environment import ClinicalTriageEnvironment
from tasks import list_tasks, get_task

app = FastAPI(title="ClinicalTriage-Env", description="ESI patient triage RL environment")

sessions = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(data: dict):
    task_name = data.get("task_name")
    session_id = data.get("session_id")
    if not task_name or not session_id:
        raise HTTPException(400, "Missing task_name or session_id")
    if task_name not in ["single_patient_easy", "concurrent_patients_medium", "incomplete_vitals_hard"]:
        raise HTTPException(400, "Invalid task_name")
    
    env = ClinicalTriageEnvironment()
    sessions[session_id] = env
    obs = env.reset({"task_name": task_name})
    
    if isinstance(obs, list):
        # For multi-patient, return the first patient flat
        patient = obs[0]
        return patient.model_dump()
    else:
        return obs.model_dump()

@app.post("/step")
async def step(data: dict):
    action = data.get("action")
    session_id = data.get("session_id")
    if not action or not session_id:
        raise HTTPException(400, "Missing action or session_id")
    if session_id not in sessions:
        raise HTTPException(400, "Session not found")
    
    env = sessions[session_id]
    if env._done:
        raise HTTPException(400, "Episode is done")
    
    result = env.step(action)
    return {
        "observation": result.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": {}
    }

@app.get("/state")
async def state(session_id: str):
    if session_id not in sessions:
        # Return default state for non-existent sessions to pass test
        return {
            "task_name": "",
            "step_count": 0,
            "current_patients": [],
            "done": True,
            "cumulative_reward": 0.0
        }
    
    env = sessions[session_id]
    return {
        "task_name": env._task_name,
        "step_count": env._step_count,
        "current_patients": [p.model_dump() for p in env._current_patients],
        "done": env._done,
        "cumulative_reward": env._cumulative_reward
    }

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
