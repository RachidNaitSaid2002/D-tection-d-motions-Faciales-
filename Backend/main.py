from fastapi import FastAPI, File, UploadFile
from DB.db import SessionLocal, Base, engine, get_db
from models.Predictions import Predictions
from schemas.prediction import PredictionCreate,PredictionResponse
import numpy as np
import sys
import os
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ML.Prediction.Pridection_func import Emotions_Predict

#creation Data base
Base.metadata.create_all(engine)

app = FastAPI()

db=SessionLocal()

@app.get("/")
def root():
    return {"message" : "Hello Worls !!!!"}


@app.post("/Prediction", response_model=PredictionResponse)
def add_Prediction(file: UploadFile = File(...)):

    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_b = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('temp.jpg', image_b)

    result = Emotions_Predict('temp.jpg')
    face_image, Label_Name, Score = result

    new_prediction = {
        "Prediction": Label_Name,
        "Score": float(Score),
        "image_path": 'default'
    }



    db_prediction = Predictions(**new_prediction)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    Result_Dir = f'../PredictiosResults/{db_prediction.id}'
    os.makedirs(Result_Dir, exist_ok=True)
    Result_Path = os.path.join(Result_Dir, 'Image.jpg')
    print(Result_Path)
    cv2.imwrite(Result_Path, face_image)

    db.query(Predictions).filter_by(id=db_prediction.id).update({"image_path": Result_Path})
    db.commit()
    db.refresh(db_prediction)

    return db_prediction


@app.get("/Prediction",response_model=list[PredictionResponse])
def get_Prediction():
    Predictions_db = db.query(Predictions).all()
    return Predictions_db

