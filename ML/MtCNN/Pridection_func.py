from mtcnn import MTCNN
from mtcnn.utils.images import load_image
import cv2
import joblib  
import numpy as np

#Creation Detector
detector = MTCNN()

def Emotions_Predict_MRCNN(Image_Path):

    Predictions = []
    
    Class_Name = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    Image = load_image(Image_Path)
    Original_Image = cv2.imread(Image_Path)

    result = detector.detect_faces(Image)


    for face in result:

        x, y, w, h = face['box']

        cv2.rectangle(Original_Image, (x, y), (x + w, y + h), (255, 255, 255), 2)

        face = Original_Image[y:y+h, x:x+w]
        
        Detection = face
        Detection_resize = cv2.resize(Detection, (32,32))
        Detection_resize = np.expand_dims(Detection_resize, axis=0)

        My_Model = joblib.load('../Model/Model.dump')
        Prediction = My_Model.predict(Detection_resize)
        Predictions.append(Prediction[0])
        Predict = list(Prediction[0])
        index_i = Predict.index(max(Predict))
        Label_Name = Class_Name[index_i]
        Score = max(Predict)
        Last_pred = (f"{Class_Name[index_i]} : {max(Predict):.2f}")

        cv2.putText(Original_Image,str(Last_pred), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 204, 255), 2, cv2.LINE_AA)
        print(Predictions)

    return Original_Image,Label_Name,Score

if __name__ == '__main__':
    Pred,_,_ = Emotions_Predict_MRCNN('../../images_Test/image copy 3.png')
    Pred = cv2.cvtColor(Pred, cv2.COLOR_BGR2RGB)
    cv2.imshow('Pred_MTCNN',Pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()