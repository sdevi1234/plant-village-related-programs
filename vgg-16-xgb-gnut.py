# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:14:07 2021

@author: Sreedevi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:30:16 2021

@author: Sreedevi
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd   
plt.style.use('dark_background')
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
 
from keras.utils import normalize, to_categorical

from keras.applications.vgg16 import VGG16
from keras.utils import generic_utils
#from keras.models import Model, Sequential
#from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential

# Read input images and assign labels based on folder names
print(os.listdir("C:/Users/Sreedevi/OneDrive/Desktop/gnut/"))

SIZE = 256  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("C:/Users/Sreedevi/OneDrive/Desktop/gnut/train/"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        #img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        #img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
       
        #Converting image to LAB Color so CLAHE can be applied to the luminance channel
        lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        #  Splitting the LAB image to L, A and B channels, respectively
        l, a, b = cv2.split(lab_img)

        plt.hist(l.flat, bins=100, range=(0,255))
        ###########Histogram Equlization#############
        #Apply histogram equalization to the L channel
        equ = cv2.equalizeHist(l)

        plt.hist(equ.flat, bins=100, range=(0,255))
        #Combine the Hist. equalized L-channel back with A and B channels
        updated_lab_img1 = cv2.merge((equ,a,b))
        
        #Convert LAB image back to color (RGB)
        hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
        
        ###########CLAHE#########################
        #Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(l)
        plt.hist(clahe_img.flat, bins=100, range=(0,255))
        
        #Combine the CLAHE enhanced L-channel back with A and B channels
        updated_lab_img2 = cv2.merge((clahe_img,a,b))
        
        #Convert LAB image back to color (RGB)
        CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
for directory_path in glob.glob("C:/Users/Sreedevi/OneDrive/Desktop/gnut/test/"):
    t_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        test_images.append(img)
        test_labels.append(t_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#############################
#Load model wothout classifier/fully connected layers
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features #This is our X input to RF
"""
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
 
import pickle
#Save model for.sav' future use
filename = 'RF_model.sav'
pickle.dump(Model, open(filename, 'wb'))


#Load model.... 
loaded_model = pickle.load(open(filename, 'rb'))

 
#Test on a different image
#READ EXTERNAL IMAGE...
test_img = cv2.imread("C:/Users/Sreedevi/OneDrive/Desktop/gnut/test/IMG-20210731-WA0023.jpg", 1)       
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
test_img = np.expand_dims(test_img, axis=0)


#predict_image = np.expand_dims(X_train[8,:,:,:], axis=0)
X_test_feature = loaded_model.predict(test_img)
X_test_feature = X_test_feature.reshape(X_test_feature.shape[0], -1)

 
#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)




#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
"""

#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_for_training, y_train) #For sklearn no one hot encoding

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction = model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction)
print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction = model.predict(input_img_features)[0] 
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])
