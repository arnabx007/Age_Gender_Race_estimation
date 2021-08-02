import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt

model = keras.models.load_model('./models/face_net.h5')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.8
fontColor              = (0,0,255)
lineType               = 1

eth_list = ['Asian', 'Black', 'Hispanic', 'Indian', 'White']
age_mapping = {0:'<10', 1:'11-15', 2:'16-20', 3:'21-25', 4:'26-30', 5:'31-35', 6:'36-42', 7:'43-50', 8:'51-60', 9:'>60'}

def main():
    gender_string, age_string, eth_string = 'Detecting..', 'Detecting..', 'Detecting..'

    cap = cv2.VideoCapture(0)
    while 1:   
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        
        i = 0
        for (x, y, w, h) in faces:
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Get the face image & resize 
            face =  gray[y:y + h, x:x + w]
            face = cv2.resize(face, dsize=(48,48), interpolation=cv2.INTER_AREA)
            face = face/255.0
            face = np.reshape(face, (1, 48,48))

            # Output of from the odel
            output = model.predict(face)
            age, gender, eth = np.argmax(output[0]), round(output[1][0][0]), np.argmax(output[2])
            
            gender_string = 'Male' if gender==0 else 'Female'
            
            age_string = f'Age: {age_mapping[age]}'

            eth_string = f'Ethnicity: {eth_list[eth]}'
            
            # The text needs to be put on every frame
            cv2.putText(img,gender_string, 
                                (x,y), 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)


            cv2.putText(img, age_string, 
                                (x+w,y+h), 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

            cv2.putText(img, eth_string, 
                                (x+w,y+h+25), 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Not more than 3 faces
            if i==3:
                break
            i+=1
        
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()