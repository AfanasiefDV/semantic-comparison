from dataclasses import dataclass
from typing import List


@dataclass
class dataAfterMatch:
    keyword: str
    isMatch: bool
    similarity: float

@dataclass
class dataRPDAfterMatch:
    text: str
    nameText: str
    keywords: List[dataAfterMatch]
