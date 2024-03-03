from dataclasses import dataclass


@dataclass
class dataAfterMatch:
    text: str
    keyword: str
    similarity: float
