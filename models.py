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
    reward: Optional[float] = None
    done: Optional[bool] = None


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
