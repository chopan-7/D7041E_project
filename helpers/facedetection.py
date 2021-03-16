import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# extractFaces returns 
def extractFaces(imageSrc, scaleFactor=1.1, minNeighbours=4):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascade_frontalface_default.xml'))
    
    # Read the input image
    img = cv2.imread(imageSrc)
    
    #cv2.imshow('Image', img)
    #cv2.waitKey()
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
    
    # list of faces
    faces_list = []
    
    for (x, y, w, h) in faces:
        crop_img = img[y:y+h, x:x+w]
        faces_list.append(crop_img)
    faces_return = np.array(faces_list)
    
    return faces_return