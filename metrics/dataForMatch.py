from dataclasses import dataclass
from typing import List


@dataclass
class dataForMatch:
    keyword: str
    isMatch: bool

@dataclass
class dataRPDForMatch:
    text: str
    nameText: str
    keywords: List[dataForMatch]
