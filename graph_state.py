from typing import TypedDict, Literal

class TriageState(TypedDict):
    review: str 
    analysis: dict
    summary: str 
    query: str 
    policy: str
    sources: list[str]
    hazard_words: list[str]
    escalate: bool
    escalated_by: Literal['rules', 'urgency', 'none', 'both']
    reply: str
    report: dict

