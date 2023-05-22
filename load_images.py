import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

# Some intializations
data_dir = os.getcwd()
train_path = data_dir + '/train'
test_path = data_dir + '/test'
labels = os.listdir(train_path)
seed = 1234
np.random.seed(seed)

# load training images and testing images
X = []; Y = []
X_test =[]; Y_test =[]
count = np.zeros((7))

for i in range(0,7):
    path = train_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        X.append(img)
        Y.append(i)
        count[i]+=1

for i in range(0,7):
    path = test_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        X_test.append(img)
        Y_test.append(i)

X = np.array(X); Y = np.array(Y)
X_test = np.array(X_test); Y_test = np.array(Y_test)

X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size = 0.25, random_state = seed, shuffle = True, stratify = Y)

def getDataset():
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def convertToRGB(dataset):
    temp = []
    for i in dataset:
        img = cv2.cvtColor(i,cv2.COLOR_GRAY2RGB)
        temp.append(img)
    return np.array(temp)

def getRGBDataset():
    X_train_new = convertToRGB(X_train)
    X_validation_new = convertToRGB(X_validation)
    X_test_new = convertToRGB(X_test)
    
    return X_train_new, Y_train, X_validation_new, Y_validation, X_test_new, Y_test
    
def getDatasetCount():
    return count