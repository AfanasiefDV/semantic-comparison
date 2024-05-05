from dataclasses import dataclass
from typing import List
from metrics.dataAfterMatch import dataAfterMatch

@dataclass
class dataRPDAfterResearch:
    text: str
    nameText: str
    keywords: List[dataAfterMatch]
    f1: List[float]
    auc: List[float]
    thresholds: List[float]
