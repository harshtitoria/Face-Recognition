import os
import cv2
import numpy as np
import pickle
from PIL import Image
path = os.path.abspath(os.path.curdir)
labels={}
x=[]
y=[]
id_=0
print(path)
for root,subdir,files in os.walk(path):
    for file in files:
        #print(file)
        if file.endswith("png") or file.endswith("jpg"):
            fpath=os.path.join(root,file)
            #print(fpath)
            pname=file.split('_')[0]
            img=Image.open(fpath).convert('L') #convert img to grayscale
            img=np.array(img,'uint8')
            # img=cv2.imread(fpath)
            # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(fpath,gray)
            if pname in labels.keys():
                x.append(img)
                y.append(labels[pname])
            else:
                labels[pname]=id_
                id_+=1
                x.append(img)
                y.append(labels[pname])

y=np.array(y)             
print(x[0])
#cv2.imshow('gray',x[0])
#cv2.waitKey(0)
print(len(x))
with open('labeldata.pkl','wb') as f:
    pickle.dump(labels,f)            #labeldata.pkl file has saved label_ids
    f.close()
rec = cv2.face.LBPHFaceRecognizer_create()
rec.train(x,y)
rec.save('ft.yml')
    #a = cv2.imread(os.path.join(subdir,file))
                                     #ft.yml is our trained recognizer
                                      
"""
use pip install opencv-contrib-python
if AttributeError: module 'cv2.cv2' has no attribute 'face'
"""