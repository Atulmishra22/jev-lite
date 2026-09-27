from typing import Annotated, Literal, Union, Any
from pydantic import BaseModel, Field

class ChoiceQuestion(BaseModel):
    type : Literal["choice"] = "choice"
    instructions : str
    criteria : dict[str, str]

class ScoreQuestion(BaseModel):
    type : Literal["score"] = "score"
    instructions : str
    levels : dict[str, str]

class NoulQuestion(BaseModel):
    type : Literal["noul"] = "noul"
    statement : str 

Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]

class SystemOneRequest(BaseModel):
    model : str = "jev-lite"
    state :  Union[str, dict[str, Any], list[Any]]
    questions : dict[str, Question]