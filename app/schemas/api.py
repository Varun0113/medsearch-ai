from typing import List, Optional

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = None
    pregnancy_status: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)
    existing_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    patient: PatientProfile = Field(default_factory=PatientProfile)


class ConditionResult(BaseModel):
    condition: str
    confidence: float
    reason: str


class MedicineResult(BaseModel):
    name: str
    type: str
    purpose: str
    safety_notes: List[str] = Field(default_factory=list)
    related_conditions: List[str] = Field(default_factory=list)
    patient_specific_warning: str = ""


class InteractionWarning(BaseModel):
    medicines: List[str] = Field(default_factory=list)
    reason: str


class QueryResponse(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    conditions: List[ConditionResult] = Field(default_factory=list)
    treatments: List[str] = Field(default_factory=list)
    otc_medicines: List[MedicineResult] = Field(default_factory=list)
    prescription_medicines: List[MedicineResult] = Field(default_factory=list)
    home_care: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    interaction_warnings: List[InteractionWarning] = Field(default_factory=list)


class SemanticQueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"


class MedicineGroup(BaseModel):
    generic_name: str
    brands: List[str] = Field(default_factory=list)
    uses: str = ""
    warnings: str = ""


class SemanticQueryResponse(BaseModel):
    query: str
    possible_conditions: List[str] = Field(default_factory=list)
    medicines: List[MedicineGroup] = Field(default_factory=list)
    home_care: List[str] = Field(default_factory=list)
