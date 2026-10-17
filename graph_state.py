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
    review_decision: Literal["approved", "edited", "not_needed"]

if __name__ == "__main__":
    from typing import get_args

    keys = TriageState.__annotations__
    print("keys:", list(keys))
    print("ok" if "review_decision" in keys else "FAIL", "review_decision present")
    print("ok" if set(get_args(keys["review_decision"])) == {"approved", "edited", "not_needed"} else "FAIL",
          "review_decision values")
    print("ok" if set(get_args(keys["escalated_by"])) == {"rules", "urgency", "both", "none"} else "FAIL",
          "escalated_by values unchanged")