import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

faceSamples=[]
Ids=[]

def getImagesAndLabels(path):
    for folder in os.listdir(path):
        newPath = path + "/" + folder

        imagePaths = [os.path.join(newPath,f) for f in os.listdir(newPath)]

        for imagePath in imagePaths:

            pilImage=Image.open(imagePath).convert('L')

            imageNp=np.array(pilImage,'uint8')

            Id=int(os.path.split(imagePath)[-1].split(".")[0])

            faces=detector.detectMultiScale(imageNp)

            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainer/trainer.yml')
