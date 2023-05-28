import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
seed = 1234
np.random.seed(seed)
# Some intializations
data_dir = os.getcwd()
train_path = data_dir + '/train'
test_path = data_dir + '/test'
affectNet_train_path =  data_dir + '/Affectnet_train' 
labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

X = []; Y = []
X_test =[]; Y_test =[]
count = np.zeros((7))

def isNormalizedIMG(img):
    """Check whether the image has been normalized before"""
    for i in img:
        for j in i:
            for k in j:
                if k > 1: return False # has not been normalized
                if k < 1 and k > 0: return True # has been normalized
    return True

# load FER2013 training dataset
for i in range(0,7):
    path = train_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        RGB_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) #convert to RGB
        if(isNormalizedIMG(RGB_img) == False): RGB_img = RGB_img/255 #if hasn't been normalized, normalize it
        X.append(RGB_img)
        Y.append(i)
        count[i]+=1
        
# load affectnet training dataset
for i in range(0,7):
    path = affectNet_train_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        resized_img = cv2.resize(img,(48,48)) #resize to (48,48,3)
        if(isNormalizedIMG(resized_img) == False): resized_img = resized_img/255 #if hasn't been normalized, normalize it
        X.append(resized_img)
        Y.append(i)
        count[i]+=1

# load FER2013 testing dataset
for i in range(0,7):
    path = test_path+'/'+labels[i]
    for j in os.listdir(path):
        img = plt.imread(path+'/'+j) #load images
        RGB_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) #convert to RGB
        if(isNormalizedIMG(RGB_img) == False): RGB_img = RGB_img/255 #If hasn't been normalized, normalize it
        X_test.append(RGB_img)
        Y_test.append(i)

# shuffle
X = np.array(X); Y = np.array(Y); X_test = np.array(X_test); Y_test = np.array(Y_test)
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size = 0.25, random_state = seed, shuffle = True, stratify = Y)

def getDataset():
    """Return np.arrays."""
    """Training set, Training set labels, Validation set, Validation set labels, Testing set, Testing set labels """
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test
    
def getDatasetCount():
    """Return count for each labels in the dataset"""
    return count