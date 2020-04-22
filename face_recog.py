import numpy as np
import cv2
import pickle
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

vid=cv2.VideoCapture(0)

recgnr = cv2.face.LBPHFaceRecognizer_create()
recgnr.read("ft.yml")
f=open('labeldata.pkl', 'rb')
labels=pickle.load(f)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,img=vid.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        facecount=int(faces.shape[0])
        simg=gray[y:y+h,x:x+w]
        id_,conf=recgnr.predict(simg)
        for i,j in labels.items():
            if j==id_:
                name=i
        #cv2.imwrite('pic.png',simg)
        cv2.putText(img,name+str(conf),(x,y+h),font,2,(255,0,0),2,cv2.LINE_4)
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(19,139,245),2)

    cv2.imshow('img',img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
