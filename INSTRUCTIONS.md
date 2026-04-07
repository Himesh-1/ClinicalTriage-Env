# ClinicalTriage-Env — Master Project Instructions

> **Meta PyTorch OpenEnv Hackathon India 2026**
> This file is the single source of truth for the entire project.
> All AI agents and contributors must follow this document exactly.

---

## Quick Reference

| Field | Value |
|---|---|
| Project Name | ClinicalTriage-Env |
| Hackathon | Meta PyTorch OpenEnv Hackathon x SST India 2026 |
| Round 1 Deadline | April 8, 2026 (online submission) |
| Finale | April 25–26, Bangalore (48hr in-person) |
| Prize Pool | $30,000 + Meta/HF interview opportunity |
| Stack | Python 3.11, FastAPI, Pydantic v2, OpenAI client, Docker, HF Spaces |
| Port | **7860** (HF Spaces requirement — never change this) |
| Build Window | 3 days |

---

## Table of Contents

1. [Hackathon Rules & Hard Requirements](#1-hackathon-rules--hard-requirements)
2. [Project Concept](#2-project-concept)
3. [Environment Design & Specification](#3-environment-design--specification)
4. [Complete File Structure](#4-complete-file-structure)
5. [File-by-File Implementation Spec](#5-file-by-file-implementation-spec)
6. [3-Day Build Plan](#6-3-day-build-plan)
7. [Final Pre-Submission Checklist](#7-final-pre-submission-checklist)
8. [Instructions for AI Agents](#8-instructions-for-ai-agents)

---

## 1. Hackathon Rules & Hard Requirements

### Official Problem Statement

> **"Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API."**

- Topic is **open** — no fixed problem statements to pick from
- Must be a **real-world task** — games and toys are explicitly disqualified
- Must implement the **full OpenEnv spec**
- Judged by programmatic checks + LLM scoring

### Submission Disqualification Checklist

Every single item below will be auto-checked. Failure on any = disqualified.

| Requirement | Notes |
|---|---|
| HF Space deploys & returns HTTP 200 | Auto-pinged on submission URL |
| `POST /reset` responds with valid observation | Must return PatientObservation JSON |
| `openenv.yaml` present and valid | Schema must match OpenEnv spec |
| Typed Pydantic models | Action, Observation, State models required |
| Dockerfile builds | Auto docker build run on submitted repo |
| `inference.py` in root directory | Must be named exactly `inference.py` |
| 3+ tasks with graders | Each grader returns float in range 0.0–1.0 |
| Baseline inference reproduces | Runs without error, produces scores |
| `[START]`/`[STEP]`/`[END]` log format exact | Any deviation = wrong eval score |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` used | Must be env vars in config |
| OpenAI client used for all LLM calls | Even if using a non-OpenAI model |
| Runtime < 20 minutes | Total inference script runtime |
| Runs on 2 vCPU / 8 GB RAM | No GPU assumptions allowed |
| Real-world task (not games/toys) | Hard disqualification rule |

---

## 2. Project Concept

### The Real-World Problem

Indian ERs handle 150M+ visits annually. A leading cause of preventable deaths is **poor triage** — the process of sorting patients by severity when they arrive. With limited staff and no clinical decision support, nurses routinely misjudge who needs attention first.

The global clinical standard is the **Emergency Severity Index (ESI)**, a 5-level triage scale. Training staff to apply ESI consistently takes months.

### The Goal

Build a fully compliant OpenEnv RL environment where an **LLM agent receives patient vitals and symptoms and must output the correct ESI triage priority level (1–5)**. The agent is graded against clinical ground truth. Over training, it learns to triage as accurately as a trained emergency nurse.

### Why This Idea Wins

- **Rule compliance** — 100% real-world task, passes the #1 disqualification check cleanly
- **Clean RL signal** — ESI scale gives objective, deterministic, programmatic ground truth
- **LLM dual scoring** — judges can score both the triage decision AND the clinical reasoning
- **Novelty** — no medical triage environment exists in the OpenEnv catalogue
- **Narrative** — "AI that saves lives in India's ERs" is unforgettable to Meta engineers

---

## 3. Environment Design & Specification

### 3.1 The ESI Triage Scale (Ground Truth)

All reward logic is grounded in this scale. Do not modify these levels.

| Level | Category | Clinical Criteria | Example |
|---|---|---|---|
| ESI-1 | Immediate / Critical | Life/limb/organ threat, requires immediate intervention | Cardiac arrest, severe respiratory distress |
| ESI-2 | Emergent / High Risk | High-risk situation, severe pain, altered mentation | Stroke symptoms, chest pain with diaphoresis |
| ESI-3 | Urgent | Stable but needs multiple resources | Moderate fever, moderate pain, possible fracture |
| ESI-4 | Less Urgent | One resource needed, stable vitals, mild symptoms | Minor laceration, ear infection |
| ESI-5 | Non-Urgent | No resources needed beyond physician exam | Prescription refill, cold symptoms |

### 3.2 Observation Space

```python
class PatientObservation(BaseModel):
    patient_id: str               # Unique ID e.g. 'P001'
    age: int                      # Patient age in years
    heart_rate: int               # BPM (normal 60–100)
    blood_pressure_systolic: int  # mmHg (normal ~120)
    blood_pressure_diastolic: int # mmHg (normal ~80)
    oxygen_saturation: float      # SpO2 % (normal >95%)
    respiratory_rate: int         # breaths/min (normal 12–20)
    temperature: float            # Celsius (normal 37.0)
    pain_scale: int               # 0–10 patient-reported
    chief_complaint: str          # Free text e.g. 'chest pain'
    arrival_mode: str             # 'walk-in' | 'ambulance'
    vitals_complete: bool         # False in Hard tasks
```

### 3.3 Action Space

```python
class TriageAction(BaseModel):
    triage_level: int    # 1–5 (ESI level)
    reasoning: str       # Clinical justification (scored by LLM grader)
    confidence: float    # 0.0–1.0 (agent's self-assessed confidence)
```

### 3.4 Reward Function

```python
def calculate_reward(action: TriageAction, ground_truth_esi: int) -> float:
    # Component 1: Exact match (worth 0.6)
    exact = 0.6 if action.triage_level == ground_truth_esi else 0.0

    # Component 2: Proximity (adjacent level = 0.3, off by 2+ = 0.0)
    if exact == 0.0:
        diff = abs(action.triage_level - ground_truth_esi)
        proximity = 0.3 if diff == 1 else 0.0
    else:
        proximity = 0.0

    # Component 3: LLM scores reasoning quality (worth up to 0.4)
    reasoning_score = llm_grade_reasoning(action.reasoning) * 0.4

    return min(1.0, (exact or proximity) + reasoning_score)
```

### 3.5 The 3 Task Levels

#### Task 1 — Easy: Single Patient, Clear Vitals

A single patient with unambiguous vitals. One correct ESI answer. Tests basic ESI knowledge.

- **Example:** 65-year-old male, BP 80/50, HR 130, O2 88%, chief complaint: "I can't breathe"
- **Ground truth:** ESI-1
- **Reward:** Exact match = up to 1.0, adjacent = up to 0.7

#### Task 2 — Medium: 5 Concurrent Patients, Must Rank All

Five patients arrive simultaneously. Agent must triage all five in order of priority. Reward based on how well the ranking matches correct ESI ordering.

- **Observation:** Five PatientObservation objects in a list
- **Ground truth:** Correct ranking order (e.g. ESI-1, ESI-2, ESI-2, ESI-3, ESI-4)
- **Reward:** Kendall tau rank correlation between agent ranking and correct ranking

#### Task 3 — Hard: Incomplete Vitals + Time Pressure

Patient data is partially missing (`vitals_complete=False`). Agent can request more info via a `request_vitals` action, but reward decays per extra step.

- **Example:** 42-year-old with chest pain, no BP reading, on beta-blockers
- **Ground truth:** ESI-2 (confirmed after requesting vitals)
- **Reward:** `base_reward - (0.05 * steps_taken)`

---

## 4. Complete File Structure

Every file listed here must exist in the final submission. Do not rename or relocate any file.

```
ClinicalTriage-Env/
├── server.py              # FastAPI app — the OpenEnv server
├── environment.py         # PatientSimulator, ESIGrader, reward logic
├── models.py              # Pydantic models: all data types
├── tasks.py               # Task definitions for Easy, Medium, Hard
├── grader.py              # Per-task grader functions returning 0.0–1.0
├── inference.py           # ← CRITICAL: Baseline LLM agent (ROOT LEVEL)
├── openenv.yaml           # ← CRITICAL: OpenEnv spec file
├── Dockerfile             # ← CRITICAL: Must build on HF Spaces
├── requirements.txt       # Python dependencies
├── README.md              # Environment description, spaces, setup
└── tests/
    └── test_compliance.py # Pre-submission validation tests
```

---

## 5. File-by-File Implementation Spec

### 5.1 `openenv.yaml`

```yaml
name: ClinicalTriage-Env
version: 1.0.0
description: >
  RL environment for AI-powered emergency room patient triage using ESI scale.
  Agent receives patient vitals and must output ESI priority level 1–5.
author: your-hf-username
license: MIT
tasks:
  - name: single_patient_easy
    difficulty: easy
    description: Triage a single patient with complete, unambiguous vitals
  - name: concurrent_patients_medium
    difficulty: medium
    description: Rank 5 concurrent patients by triage priority
  - name: incomplete_vitals_hard
    difficulty: hard
    description: Triage with missing data and time-decay reward
action_space: TriageAction
observation_space: PatientObservation
reward_range: [0.0, 1.0]
```

### 5.2 `models.py`

```python
from pydantic import BaseModel
from typing import Optional, List

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
    arrival_mode: str             # 'walk-in' | 'ambulance'
    vitals_complete: bool

class TriageAction(BaseModel):
    triage_level: int             # 1–5
    reasoning: str
    confidence: float

class EnvironmentState(BaseModel):
    task_name: str
    step_count: int
    current_patients: List[PatientObservation]
    done: bool
    cumulative_reward: float

class StepResult(BaseModel):
    observation: PatientObservation
    reward: float
    done: bool
    info: dict
```

### 5.3 `server.py`

Must expose exactly these endpoints:

```
POST /reset   → accepts optional task_name param, returns PatientObservation
POST /step    → accepts TriageAction, returns StepResult
GET  /state   → returns EnvironmentState
GET  /health  → returns {"status": "ok"}  ← used by HF auto-ping
GET  /tasks   → returns list of available task names
```

Critical rules:
- Use **FastAPI only** — no Flask, no Starlette bare
- Keep session state in a dict keyed by `session_id` (UUID)
- Do NOT hardcode patient data — use `PatientSimulator` from `environment.py`
- All endpoints must handle exceptions and return proper HTTP status codes

### 5.4 `environment.py`

- `PatientSimulator.generate(difficulty)` → returns a `PatientObservation`
- `ESIGrader.grade(action, ground_truth, step_count)` → returns float 0.0–1.0
- Use `numpy` for random generation, **seed with 42** for reproducibility
- Ground truth ESI must be **deterministically derived from vitals**, not random
- Store a `patient_id → ground_truth_esi` mapping in session state

### 5.5 `inference.py` ⚠️ HIGHEST RISK FILE

> This file MUST be in root. MUST use OpenAI client. MUST output logs in exact `[START]`/`[STEP]`/`[END]` format. Any deviation causes scoring failure.

**Required log format — must match exactly:**

```python
# Start of each task episode
print(f'[START] task={task_name} episode={ep_num}')

# Each step
print(f'[STEP] step={step} action={action_json} reward={reward:.4f}')

# End of each episode
print(f'[END] task={task_name} episode={ep_num} total_reward={total:.4f} steps={steps}')
```

**Environment variable usage:**

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ['API_BASE_URL'],
    api_key=os.environ['HF_TOKEN']
)
MODEL = os.environ['MODEL_NAME']

# All LLM calls MUST use this client — no other LLM libraries allowed
```

### 5.6 `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

> **HF Spaces requires port 7860. Do not change this.**

### 5.7 `requirements.txt`

```
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
numpy>=1.26.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

---

## 6. 3-Day Build Plan

> Complete each day fully before moving to the next. Jumping ahead causes cascading failures.

### Day 1 — Core Environment

Goal by end of day: server runs locally, all 3 tasks can be stepped through manually.

1. Set up repo, folder structure, install dependencies
2. Write `models.py` — all Pydantic models
3. Write `tasks.py` — 3 task configs with patient profiles and ground truth ESI
4. Write `environment.py` — `PatientSimulator` and `ESIGrader`
5. Write `server.py` — FastAPI app with all endpoints
6. Test: `curl http://localhost:7860/reset` → should return a patient JSON
7. Test: `curl -X POST /step` with a `TriageAction` → should return `StepResult` with reward

### Day 2 — Docker + HF Spaces Deploy

Goal by end of day: HF Space is live, `reset()` and `step()` work from the public URL.

1. Write `openenv.yaml` with all required fields
2. Write `Dockerfile` (port 7860, python:3.11-slim)
3. Test: `docker build . && docker run -p 7860:7860` → server must start
4. Create HF Space (Docker SDK type), push repo
5. Test: `HF_SPACE_URL/health` returns 200
6. Test: `HF_SPACE_URL/reset` returns valid `PatientObservation`
7. Write `tests/test_compliance.py`, run pre-submission validator
8. Fix any issues found

### Day 3 — inference.py + README + Submit

Goal by end of day: submitted.

1. Write `inference.py` — LLM agent that calls `reset()` then loops `step()` for all 3 tasks
2. Verify **exact** `[START]`/`[STEP]`/`[END]` log format (copy from dashboard sample)
3. Test: `inference.py` completes in < 20 min on standard machine
4. Confirm all 3 env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) are used via OpenAI client
5. Run pre-submission validation script from the dashboard
6. Write `README.md` — environment description, action/observation spaces, setup steps
7. Go through every item in the checklist below
8. Paste HF Spaces URL on dashboard before deadline

---

## 7. Final Pre-Submission Checklist

Go through every item before submitting. Check them off one by one.

- [ ] HF Space URL is publicly accessible (open in incognito)
- [ ] `GET /health` returns HTTP 200 and `{"status": "ok"}`
- [ ] `POST /reset` returns valid `PatientObservation` JSON
- [ ] `POST /step` with valid action returns `StepResult`
- [ ] `GET /state` returns `EnvironmentState`
- [ ] `openenv.yaml` exists and has all required fields
- [ ] All 3 tasks listed in `openenv.yaml` under `tasks:`
- [ ] `Dockerfile` builds without errors (`docker build .`)
- [ ] `inference.py` is in the **root directory** (not a subfolder)
- [ ] `inference.py` uses OpenAI client (`from openai import OpenAI`)
- [ ] `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are all used in `inference.py`
- [ ] `[START]`/`[STEP]`/`[END]` log format is exactly correct (run and inspect stdout)
- [ ] `inference.py` completes in under 20 minutes (`time python inference.py`)
- [ ] All reward values are floats in range 0.0–1.0
- [ ] `README.md` documents action and observation spaces
- [ ] Pre-submission validation script passes all checks

---

## 8. Instructions for AI Agents

### What You May and May Not Change

**You MAY change / improve:**
- Patient simulation logic and scenario variety in `environment.py`
- Number of patient scenarios per difficulty level
- Reward formula weights (stay within 0.0–1.0 total)
- README content and formatting
- LLM prompt engineering in `inference.py`

**NEVER change these:**
- File names — especially `inference.py`
- Port number — must stay **7860**
- Log format — `[START]`/`[STEP]`/`[END]` exact field names and order
- Endpoint paths — `/reset`, `/step`, `/state`, `/health` must stay exactly as specified
- Framework — FastAPI only
- LLM client — OpenAI client only, no other libraries
- ESI scale — levels 1–5 must stay intact
- The domain — must remain real-world medical triage, not a game or toy

### Bootstrap Prompt (Paste This Into Any New AI Session)

```
You are working on ClinicalTriage-Env, an OpenEnv RL environment for the
Meta PyTorch OpenEnv Hackathon India 2026. The environment simulates emergency
room patient triage using the ESI scale (1=Critical to 5=Non-urgent).

Tech stack: Python 3.11, FastAPI, Pydantic v2, OpenAI client, Docker, HF Spaces.
Port: 7860 (required by HF Spaces — never change).
Framework: FastAPI only.

Hackathon hard rules:
- inference.py MUST be in root, use OpenAI client, output exact [START]/[STEP]/[END] logs
- API_BASE_URL, MODEL_NAME, HF_TOKEN are the only allowed env vars for LLM calls
- 3 tasks: single_patient_easy, concurrent_patients_medium, incomplete_vitals_hard
- Rewards are floats 0.0–1.0. Runtime < 20min. Runs on 2vCPU / 8GB RAM.
- Real-world task — NOT a game or toy (hard disqualification rule).

Refer to INSTRUCTIONS.md for full specs before writing any code.
Do not change file names, port, log format, or endpoint paths.
```

---

*ClinicalTriage-Env | Meta OpenEnv Hackathon India 2026 | Master Project Document*
