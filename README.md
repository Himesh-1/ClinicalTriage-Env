---
title: ClinicalTriage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# ClinicalTriage-Env

> **Meta PyTorch OpenEnv Hackathon x SST India 2026**

An OpenEnv-compliant reinforcement learning environment for AI-powered emergency room patient triage using the **Emergency Severity Index (ESI)** scale.

---

## Installation

```bash
pip install clinicaltriage-env
```

## Quick start — client SDK

```python
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
```

## Evaluation harness

```bash
# Run 5 episodes per task locally (no server needed):
python eval/benchmark.py --local --episodes 5

# Run against a deployed server:
python eval/benchmark.py --base-url ws://localhost:7860 --episodes 5
```

---

## The Problem

Indian ERs handle 150M+ visits annually. A leading cause of preventable deaths is **poor triage** — the process of sorting patients by severity when they arrive. With limited staff and no clinical decision support, nurses routinely misjudge who needs attention first.

The global clinical standard is the **Emergency Severity Index (ESI)**, a 5-level triage scale. This environment trains an LLM agent to triage patients as accurately as a trained emergency nurse.

---

## Environment Overview

| Field | Value |
|---|---|
| **Port** | 7860 (HF Spaces requirement) |
| **Framework** | FastAPI |
| **Reward Range** | [0.0, 1.0] |
| **Tasks** | 3 (Easy, Medium, Hard) |
| **Stack** | Python 3.11, FastAPI, Pydantic v2, OpenAI client |

---

## Observation Space (`PatientObservation`)

```json
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
  "vitals_complete": true
}
```

| Field | Type | Description |
|---|---|---|
| `patient_id` | str | Unique patient identifier |
| `age` | int | Patient age in years |
| `heart_rate` | int | BPM (normal 60–100) |
| `blood_pressure_systolic` | int | mmHg (normal ~120) |
| `blood_pressure_diastolic` | int | mmHg (normal ~80) |
| `oxygen_saturation` | float | SpO2 % (normal >95%) |
| `respiratory_rate` | int | breaths/min (normal 12–20) |
| `temperature` | float | Celsius (normal 37.0) |
| `pain_scale` | int | 0–10 patient-reported |
| `chief_complaint` | str | Free text |
| `arrival_mode` | str | `'walk-in'` or `'ambulance'` |
| `vitals_complete` | bool | `False` in Hard task |

---

## Action Space (`TriageAction`)

```json
{
  "triage_level": 1,
  "reasoning": "Critical patient with SpO2 88%, HR 130, BP 80/50 — immediate ESI-1 intervention required.",
  "confidence": 0.95
}
```

| Field | Type | Description |
|---|---|---|
| `triage_level` | int (1–5) | ESI level (1=Critical, 5=Non-urgent). Use **0** in Hard task to request vitals. |
| `reasoning` | str | Clinical justification (scored by LLM grader) |
| `confidence` | float (0–1) | Agent self-assessed confidence |

---

## ESI Triage Scale (Ground Truth)

| Level | Category | Clinical Criteria |
|---|---|---|
| **ESI-1** | Immediate / Critical | Life/limb/organ threat |
| **ESI-2** | Emergent / High Risk | High-risk, severe pain, altered mentation |
| **ESI-3** | Urgent | Stable but needs multiple resources |
| **ESI-4** | Less Urgent | One resource needed, mild symptoms |
| **ESI-5** | Non-Urgent | Physician exam only |

---

## Tasks

### Task 1 — Easy: Single Patient, Clear Vitals
Single patient with unambiguous vitals. Reward based on exact ESI match + reasoning quality.

### Task 2 — Medium: 5 Concurrent Patients
Five patients arrive simultaneously. Agent must rank all five by priority.
- Reward = Kendall Tau rank correlation between agent ranking and correct ranking.
- Encode ranking as a JSON array of patient IDs in the `reasoning` field.

### Task 3 — Hard: Incomplete Vitals + Time Pressure
Patient data is partially missing. Agent can request more info (set `triage_level=0`) but reward decays by 0.05 per extra step.

---

## Reward Function

```
reward = exact_match (0.6) + proximity (0.3) + reasoning_quality (0.4) - step_penalty
```

- **Exact match**: 0.6 if `triage_level == ground_truth_esi`
- **Proximity**: 0.3 if off by 1 level (no exact match)
- **Reasoning**: LLM-graded clinical quality × 0.4
- **Step penalty** (Hard only): −0.05 per step beyond the first

All rewards are clamped to **[0.0, 1.0]**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment, returns `PatientObservation` |
| `POST` | `/step` | Submit `TriageAction`, returns `StepResult` |
| `GET` | `/state` | Returns `EnvironmentState` |
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/tasks` | Returns list of available tasks |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "single_patient_easy"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "triage_level": 1,
      "reasoning": "Critical: BP 80/50, SpO2 88%, HR 130 — ESI-1.",
      "confidence": 0.95
    }
  }'
```

---

## Setup & Local Development

### Prerequisites
- Python 3.11
- Docker (for container testing)

### 1. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 2. Run the inference agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

---

## Docker

```bash
# Build
docker build -t clinicaltriage-env .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="..." \
  -e MODEL_NAME="..." \
  -e HF_TOKEN="..." \
  clinicaltriage-env
```

---

## Environment Variables

| Variable         | Required | Default                          | Description                    |
| ---------------- | -------- | -------------------------------- | ------------------------------ |
| API_BASE_URL     | Yes      | https://router.huggingface.co/v1 | LLM API endpoint               |
| MODEL_NAME       | Yes      | meta-llama/Llama-3.1-8B-Instruct | Model identifier               |
| HF_TOKEN         | Yes      | none                             | HuggingFace / API key          |
| ENV_BASE_URL     | No       | http://localhost:7860            | Running environment server URL |
| LOCAL_IMAGE_NAME | No       | clinicaltriage-env:latest        | Docker image name              |
| SKIP_LLM_GRADER  | No       | false                            | Skip LLM judge (local testing) |

---

## License

MIT

---

*ClinicalTriage-Env | Meta PyTorch OpenEnv Hackathon x SST India 2026*
