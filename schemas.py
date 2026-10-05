from pydantic import BaseModel, Field
from typing import Literal, Optional

class ReviewAnalysis(BaseModel):

    product: str = Field(..., description="The product mentioned in the review")
    sentiment: Literal["positive", "negative", "mixed"] = Field(..., description="The sentiment of the user review, and mixed if you are unsure whether the review is positive or negative. Use 'mixed' when the review contains both praise and a complaint. Use 'negative' only when there is no praise at all.")
    issues: Optional[list[str]] = Field(default = None, description="Any issues mentioned in the review. Return an empty list if there are no issues." )
    urgency: Literal['low','medium','high',"none"] = Field(..., description=(
        "How urgently a person needs to respond to this review. "
        "'high': the review describes a safety hazard (burning, smoke, sparks, overheating, or a risk of injury), "
        "money that is owed or has been withheld, a problem that support has already failed to resolve after the customer contacted them, "
        "or a threat of legal action or public complaint. "
        "'medium': a faulty or damaged product, or a late delivery, with none of the 'high' conditions. "
        "'low': a minor annoyance, a matter of preference, or a question. "
        "'none': a positive review with no problem. "
        "Asking for a return, refund or replacement does not by itself make a review 'high'. "
        "Strong or angry wording does not by itself make a review 'high'; judge by what happened, not by the tone."
    ))    
    reviewer: Optional[str] = Field(default=None, description="The name of the person who is making the review")



