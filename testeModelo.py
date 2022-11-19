import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp

face = mp.solutions.face_detection
Face = face.FaceDetection()
mpDwaw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

model = load_model('modelo.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'SEM SORRISO'
    elif classNo == 1:
        return 'COM SORRISO'

while True:
    success, imgOrignal = cap.read()

    imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
    results = Face.process(imgRGB)
    facesPoints = results.detections
    hO, wO, _ = imgRGB.shape
    if facesPoints:
        for id, detection in enumerate(facesPoints):
            #mpDwaw.draw_detection(img, detection)
            bbox = detection.location_data.relative_bounding_box
            x,y,w,h = int(bbox.xmin*wO),int(bbox.ymin*hO),int(bbox.width*wO),int(bbox.height*hO)
            imagemFace = imgOrignal[y-30:y+h,x:x+w]
            img = np.asarray(imagemFace)
            img = cv2.resize(img, (64, 64))
            img = preprocessing(img)
            img = img.reshape(1, 64, 64, 1)

            predictions = model.predict(img)
            indexVal = np.argmax(predictions)
            probabilityValue = np.amax(predictions)
            print(indexVal,probabilityValue)
            if indexVal ==0:
                cor = (0,0,255)
            else:
                cor = (0,255,0)

            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), cor, 3)

            cv2.putText(imgOrignal, str(getCalssName(indexVal)), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        cor, 8, cv2.LINE_AA)

            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cor, 2,
                        cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    #cv2.imshow("Face", imagemFace)
    cv2.waitKey(1)

