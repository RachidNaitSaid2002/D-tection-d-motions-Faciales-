from sqlalchemy import  Integer, String, Float, Column, JSON
from DB.db import Base

class Predictions(Base):
    __tablename__ = 'Predictions'

    id = Column(Integer, primary_key=True)
    Prediction = Column(String)
    Score = Column(Float)
    image_path = Column(String, nullable=False)