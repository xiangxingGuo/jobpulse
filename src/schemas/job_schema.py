from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


RoleCategory = Literal["MLE","DS","DE","SWE","RE","Other"]
Seniority = Literal["Intern","NewGrad","Junior","Mid","Senior","Staff","Other"]
WorkMode = Literal["Remote","Hybrid","Onsite","Unknown"]


class VisaInfo(BaseModel):
    requires_us_auth: bool = False
    opt_cpt_ok: bool = False
    sponsorship_mentioned: bool = False


class JobStructured(BaseModel):
    role_category: RoleCategory = "Other"
    seniority: Seniority = "Other"
    work_mode: WorkMode = "Unknown"
    location: Optional[str] = None

    visa: VisaInfo = Field(default_factory=VisaInfo)

    skills: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)

    years_required: Optional[float] = None
    confidence: float = 0.5
