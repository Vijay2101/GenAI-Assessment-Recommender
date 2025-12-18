from typing import List
from pydantic import BaseModel


class RecommendedAssessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]
