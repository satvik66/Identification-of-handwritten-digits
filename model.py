import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
import random
import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from PIL import Image


(trainX, trainy), (testX, testy) = mnist.load_data()

train = np.concatenate((trainX,testX)) 
test = np.concatenate((trainy,testy)) 
print(len(train))
print(len(test))

data_train, data_test, target_train, target_test =  train_test_split(train,test,stratify=test, test_size=0.25,random_state=42)
#train_test_split(train,test, test_size=0.25,random_state=42)

print(len(data_train),len(target_train))
print(len(data_test),len(target_test))

print(data_test[2])

plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger

for i in range(9):
    plt.subplot(3,3,i+1)
    r = random.randint(0, len(data_train))
    plt.imshow(data_train[r], cmap='gray', interpolation='none')
    plt.title("Class {}".format(target_train[r]))
    
plt.tight_layout()

# just a little function for pretty printing a matrix
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  
matprint(data_train[r])

data_train.shape[0]

def preproc(data_train,data_test):
  data_train = data_train.reshape((data_train.shape[0], 28, 28, 1))
  data_test = data_test.reshape((data_test.shape[0], 28, 28, 1))
  print(data_test.shape)

  data_train = data_train.astype('float32')   # change integers to 32-bit floating point numbers
  data_test = data_test.astype('float32')

  data_train /= 255.0                       # normalize each value for each pixel for the entire vector for each input
  data_test /= 255.0
  return data_train, data_test
data_train,data_test= preproc(data_train,data_test)
print("Training matrix shape", data_train.shape)
print("Testing matrix shape", data_test.shape)

classes =len(np.unique(target_train))
y_train = to_categorical(target_train,classes)
y_test = to_categorical(target_test,classes)

model = keras.Sequential(
    [
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(classes, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 128
epochs = 15

model.fit(data_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(data_test,y_test)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

predictions = model.predict(data_test)

predictions
prediction = []
for i in predictions:
    prediction.append(np.argmax(i))
print(prediction)

print(target_test)

model.save("my_project_model.h5")
