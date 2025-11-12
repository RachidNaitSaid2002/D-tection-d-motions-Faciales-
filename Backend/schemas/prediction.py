from pydantic import BaseModel

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