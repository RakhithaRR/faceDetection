import cv2
import numpy as np

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainer/trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

database = open("database.txt", "r")
users = [line.rstrip() for line in database]
print users


cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_DUPLEX, 1, 1, 0, 1, 1)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if(confidence < 100):
            user = users[Id-1].split(".")[1]
        else:
            user="Unknown"

        cv2.cv.PutText(cv2.cv.fromarray(im),str(user), (x,y),font, (255,255,0))

    cv2.imshow('Recognizer',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

