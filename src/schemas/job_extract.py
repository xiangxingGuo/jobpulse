from pydantic import BaseModel, Field
from typing import Optional, List


class JobExtract(BaseModel):
    """
    Canonical structured representation of a job description.
    This is the contract between LLM extraction and downstream analytics.
    """

    role_title: Optional[str] = None
    company: Optional[str] = None

    location: Optional[str] = None
    employment_type: Optional[str] = None  # Internship, Full-time, Contract, etc.
    remote_policy: Optional[str] = None  # Remote, Hybrid, Onsite

    responsibilities: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    preferred_qualifications: List[str] = Field(default_factory=list)

    skills: List[str] = Field(default_factory=list)

    years_experience_min: Optional[int] = None
    degree_level: Optional[str] = None  # BS, MS, PhD, Any, None

    visa_sponsorship: Optional[str] = None  # Yes / No / Unclear
