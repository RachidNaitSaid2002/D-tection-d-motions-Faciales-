import joblib
import cv2
import numpy as np



def Emotions_Predict(Myimage):

    
    Class_Name = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    Face_Cascade = cv2.CascadeClassifier('../ML/Haarcascade/haarcascade_frontalface_default.xml')

    if Face_Cascade.empty():
        print('Script Not Loaded !!')

    else :
        Test_Image = cv2.imread(Myimage)

        face_image = Test_Image.copy()

        face_rect = Face_Cascade.detectMultiScale(face_image, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in face_rect:

            cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

            face = face_image[y:y+h, x:x+w]
        
            Detection = face
            Detection_resize = cv2.resize(Detection, (32,32))
            Detection_resize = np.expand_dims(Detection_resize, axis=0)


            My_Model = joblib.load('../ML/Model/Model.dump')
            Prediction = My_Model.predict(Detection_resize)
            Predict = list(Prediction[0])
            index_i = Predict.index(max(Predict))
            Label_Name = Class_Name[index_i]
            Score = max(Predict)
            Last_pred = (f"{Class_Name[index_i]} : {max(Predict):.2f}")

            cv2.putText(face_image,str(Last_pred), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 204, 255), 2, cv2.LINE_AA)

        return face_image,Label_Name,Score

if __name__ == '__main__':
    Pred,_,_ = Emotions_Predict('../../images_Test/image copy 6.png')
    cv2.imshow('Pred',Pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()