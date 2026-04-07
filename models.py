from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Observation(BaseModel):
    num_rows: int
    num_columns: int
    missing_ratio: float
    outlier_ratio: float
    skewness: List[float]
    data_types: List[str]
    steps_remaining: int

    reward: float = 0.0
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class Action(BaseModel):
    action_type: str
    reason: str
    confidence: float

    class Config:
        extra = "allow"