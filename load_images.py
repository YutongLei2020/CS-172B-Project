import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Some intializations
data_dir = 'archive/'
train_path = data_dir + 'train/'
test_path = data_dir + 'test/'
labels = os.listdir(train_path)
seed = 1234
np.random.seed(seed)

# load training images and testing images
X = []; Y = []
X_test =[]; Y_test =[]

for i in range(0,7):
    path = train_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        X.append(img)
        Y.append(i)

for i in range(0,7):
    path = test_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        X_test.append(img)
        Y_test.append(i)

X = np.array(X); Y = np.array(Y)
X_test = np.array(X_test); Y_test = np.array(Y_test)

# shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
new_X, new_Y = unison_shuffled_copies(X,Y)

# splitting: 75% training data + 25% validation data
index = len(new_X)//4 #28709//4 = 7177
X_train = new_X[index:]
Y_train = new_Y[index:]
X_validation = new_X[:index]
Y_validation = new_Y[:index]

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
    
    