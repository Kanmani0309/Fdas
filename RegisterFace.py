import cv2
import time
from TrainFace import trainface

vidcap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("This is the Capture Face and Train Program!!! ")
Id = input('Enter your id - ')
sampleCntr = 0
while (True):
    ret, img = vidcap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        time.sleep(0.5)

        sampleCntr = sampleCntr + 1
        # saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User." + Id + '.' + str(sampleCntr) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # Collect 10 samples
    elif sampleCntr > 11:
        break
vidcap.release()
cv2.destroyAllWindows()
time.sleep(1)
print("Now training the model .... ")
trainface()