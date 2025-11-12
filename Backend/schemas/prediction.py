from pydantic import BaseModel
from typing import List

class PredictionBase(BaseModel):
    Prediction : str
    Score : float
    image_path : str

class PredictionCreate(PredictionBase):
    pass

class PredictionResponse(PredictionBase):
    id : int

    class Config:
        from_attributes = True