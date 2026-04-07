from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


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
    arrival_mode: str  # 'walk-in' | 'ambulance'
    vitals_complete: bool


class TriageAction(BaseModel):
    triage_level: int = Field(..., ge=0, le=5, description="ESI level 1–5 (0 = request_vitals in Hard task)")
    reasoning: str = Field(..., description="Clinical justification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent self-assessed confidence")


class EnvironmentState(BaseModel):
    task_name: str
    step_count: int
    current_patients: List[PatientObservation]
    done: bool
    cumulative_reward: float


class StepResult(BaseModel):
    observation: Optional[PatientObservation]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_name: Optional[str] = "single_patient_easy"
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action: TriageAction
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str
