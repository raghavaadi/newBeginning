from pydantic import BaseModel
from typing import List

class VideoAnalysis(BaseModel):
    analysis: dict
    transcript: str
    clips: List[dict]