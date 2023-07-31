import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#this is taking me to the direcotry where our file ,this file,is faces-train.py
image_dir = os.path.join(BASE_DIR, "images")
#this is looking for the "images" folder to look for training imagess

#face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

face_cascade = cv2.CascadeClassifier(r'C:\IIT Spring 23\Internet of Things\Facial_Recognition Part2\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#function in the OpenCV that creates an instance of a
#  face recognizer object using a Local Binary Patterns Histograms(LBPH) algorithm. 

#This algorithm extracts features from the image and then use a machine learning algorithm and does the class
#   ification of these features as belonging to respective individuals. 
current_id = 0  #intitalize
label_ids = {}
y_labels = []
x_train = [] #initialized

#training is done on the image dataset
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file) 
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550) 
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #this converts the image  into array numbers values that will use to train
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            #adjusting the scale factors and minneighbours to get the best tradeoff for accuracy 
            #and detection speed
            #selecting 1.5 scalefactor gives the best combination after trial and error method
            #for me changing min neighbour to anyother number ,the detection rate reduced

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

#saving the trained model in pickle file
with open('pickles/pickles.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

#
recognizer.train(x_train, np.array(y_labels))
recognizer.save('recognizers/trainer.yml')

