from typing import Optional,List
from pydantic import BaseModel, Field

class RecommendedAssessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int] = None
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]


class HiringIntent(BaseModel):
    skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    job_level: Optional[str] = None
    required_test_types: List[str] = Field(default_factory=list)
