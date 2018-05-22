import cv2
import os

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

Id = raw_input('Enter your ID: ')
name = raw_input('Enter your name: ')
sampleNum = 0

db = open("database.txt","a")
db.write(Id + "." + name + "\n")
db.close()

while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
        )

        for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                sampleNum += 1
                pathFolder = "dataSet/" + name
                if not os.path.exists(pathFolder):
                        os.makedirs(pathFolder)
                cv2.imwrite(pathFolder + "/" + Id + "." + str(sampleNum)+ ".jpg", gray[y:y+h,x:x+w])


                cv2.imshow('frame', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        elif sampleNum > 20:
                break

cap.release()
cv2.destroyAllWindows()
