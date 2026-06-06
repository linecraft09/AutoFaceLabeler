from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FacialFeatures(BaseModel):
    age: int
    race: str
    hair_color: str
    confidence: Dict[str, Any]


class FacialMotion(BaseModel):
    description: str
    key_movements: List[str]
    duration_category: str


class ClarityScore(BaseModel):
    mean_clarity: float
    median_clarity: float
    std_clarity: float
    min_clarity: float
    max_clarity: float
    face_detected_ratio: float
    per_frame: Optional[List[float]] = None


class ExpressionIntensity(BaseModel):
    intensity: int = Field(..., ge=1, le=5)
    rationale: str
    dominant_expressions: List[str]


class Transcription(BaseModel):
    text: str
    language: str
    segments: List[Dict[str, Any]]


class AnnotationResult(BaseModel):
    video_id: str
    platform: str
    facial_features: Optional[FacialFeatures] = None
    facial_motion: Optional[FacialMotion] = None
    clarity: Optional[ClarityScore] = None
    expression: Optional[ExpressionIntensity] = None
    transcription: Optional[Transcription] = None
