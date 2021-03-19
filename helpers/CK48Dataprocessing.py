import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from time import sleep
from tqdm import tqdm

def loadDataset(path):
    files=os.listdir(path)
#     print(files)

    Exp=['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    i=0
    last=[]
    images=[]
    labels=[]
    images_o=[]
    
    for fle in files:
        idx=Exp.index(fle)
        label=idx
        total=path+'/'+fle
        files_exp= os.listdir(total)
        
        for fle_2 in tqdm(range(len(files_exp)), ncols=100, desc=f"Importing from {fle}"):
            file_main=total+'/'+files_exp[fle_2]
#             print(file_main+"   "+str(label))

            image= cv2.imread(file_main)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_o.append(image)
            image= cv2.resize(image,(48,48))
            images.append(image)
            labels.append(label)
            i+=1
        last.append(i)
    images_f=np.array(images)
    images_f_2=images_f/255
    labels_f=np.array(labels)
    
    return images_f_2, images_f, labels_f, images, images_o, labels, Exp

def TrainTestSplit(images, labels, testSize=0.25):
    # encode the labels
    num_of_classes = len(np.unique(labels))
    labels_encoded = tf.keras.utils.to_categorical(labels, num_classes=num_of_classes)
    
    # split dataset
    X_train, X_test, Y_train, Y_test= train_test_split(images, labels_encoded, test_size=testSize)
    
    return X_train, X_test, Y_train, Y_test
    
    