from pydantic import BaseModel, Field

class QualityScore(BaseModel):
    """Data model for an album."""

    score: float
    feedback: str
