import cv2
import os

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0)
i=1
while i<101:
    ret,frame=cam.read()
    if ret:
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, 1.3, 5)
        for x,y,w,h in faces:
            cv2.imwrite('dataset/rohan/Rohan_'+str(i)+'.jpg',gray[y:y+h,x:x+w])
        i+=1   
    else:
        print('camera not working')
cam.release()
cv2.destroyAllWindows()             
