from typing import List
from enum import IntEnum,Enum
from pydantic import BaseModel, Field # type: ignore

class OneFourScore(IntEnum):
    """Score enumeration for judgment tasks."""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

class LikertScore(IntEnum):
    """Score enumeration for judgment tasks based on likert score."""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    
class FitScore(Enum):
    """Score enumeration for judgment tasks."""
    BESTFIT = 'best_fit'
    GOODFIT = 'good_fit'
    BADFIT = 'bad_fit'

class BoolScore(Enum):
    """True/False judgment results."""
    TRUE = True
    FALSE = False

class BorderlineScore(Enum):
    """Score enumeration for judgment tasks."""
    PASS = "Pass"
    BORDERLINE = "Borderline"
    FAIL = "Fail"

class LLMJudgeBool(BaseModel):
    """Judgment result model for detailed output."""
    reasoning: List[str] = Field(
        description="A list of strings explaining the reasoning behind the judgment."
    )
    total_score: BoolScore = Field(
        description="The final judgment score or a binary True/False decision."
    )

class LLMJudgeOneFour(BaseModel):
    reasoning: List[str] = Field(
        description="A list of strings explaining the reasoning behind the judgement."
    )
    total_score: OneFourScore = Field(
        description="The total score from the judgement from these 4 options : 4-totally acceptable 3-mostly acceptable 2-mostly unacceptable 1-totally unacceptable.."
    )

class LLMJudgeFit(BaseModel):
    reasoning: List[str] = Field(
        description="A list of strings explaining the reasoning behind the judgement."
    )
    total_score: FitScore = Field(
        description="The total score from the judgement from these 3 options :   (Best Fit),  (Good Fit), (Poor Fit)."
    )    



class LLMJudgeBorderline(BaseModel):
    reasoning: List[str] = Field(
        description="A list of strings explaining the reasoning behind the judgement."
    )
    total_score: BorderlineScore = Field(
        description="The total score from the judgement from these 3 options :   Pass (acceptable), Borderline (marginal, needs revision), Fail (unacceptable)."
    )    

class LLMJudgeLikert(BaseModel):
    reasoning: List[str] = Field(
        description="A list of strings explaining the reasoning behind the judgement."
    )
    total_score: LikertScore = Field(
        description="The total score from the judgement from these 3 options based on likert scale :   1 (Strongly Disagree), 2 (Disagree), 3 (Neutral), 4 (Agree), 5 (Strongly Agree)."
    )    