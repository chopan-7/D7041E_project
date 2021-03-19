import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm


def UTKFaceProcessing(fldr):
    #read files from directory
    files = os.listdir(fldr)
    
    ages = []
    genders = []
    images = []
    labels = []
    
    images_o = []
    
    # importing images
    for fle in tqdm(range(len(files)), ncols=100, desc="Importing images"):
        age = int(files[fle].split('_')[0])
        gender = int(files[fle].split('_')[1])
        total = fldr + '/' + files[fle]
        
        image = cv.imread(total)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        images_o.append(image)
        image = cv.resize(image, (48,48))
        images.append(image)
    
    # importing labels
    for fle in tqdm(range(len(files)), ncols=100, desc="Importing labels"):
        label = []
        age = int(files[fle].split('_')[0])
        gender = int(files[fle].split('_')[1])
        label.append(age)
        label.append(gender)
        labels.append(label)
        
        ages.append(age)
        genders.append(gender)
     
    return images, labels, ages, genders, images_o

def UTKFaceTrainTestSplit(images, labels, testSize=0.2):
    images_f = np.array(images)/255
    labels_f = np.array(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(images_f, labels_f, test_size=testSize)
    return X_train, X_test, Y_train, Y_test

def UTKFaceDataDistribution(genders, ages):
    genders_f =np.array(genders)
    ages_f = np.array(ages)
    
    gender_val, gender_counts = np.unique(genders_f, return_counts=True)
    ages_val, ages_counts = np.unique(ages_f, return_counts=True)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    gender =['Male', 'Female']
    values = gender_counts[:2]
    ax.bar(gender,values)
    plt.ylabel('distribution')
    plt.show()
    
    plt.plot(ages_counts)
    plt.xlabel('ages')
    plt.ylabel('distribution')
    plt.show()
    