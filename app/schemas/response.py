from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

class ChoiceAnswer(BaseModel):
    type : Literal["choice"] = "choice"
    choice : str
    probabilities : dict[ str , float] 
    confidence : float

class ScoreAnswer(BaseModel):
    type : Literal["score"] = "score"
    score : float
    probabilities : dict[str, float]
    confidence : float

class NoulAnswer(BaseModel):
    type : Literal["noul"] = "noul"
    noul : float

Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer], Field(discriminator="type")]

class SystemOneResponse(BaseModel):
    model : str
    answers : dict[str, Answer]