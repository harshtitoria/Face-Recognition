# face-recognition
OpenCV project to recognize face in real time using webcam.
The face_data.py file capture 100 images of the person, the face_train.py file train the model on these faces
to create labeldata.pkl which has saved label_ids and ft.yml which is our trained recognizer. The face_recog.py uses
these two files to recognize the person.
we used LBPHFaceRecognizer of OpenCV and haarcascade_frontalface_default.xml for this project.  
