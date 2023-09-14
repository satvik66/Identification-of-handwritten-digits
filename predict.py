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


app = Flask(__name__, template_folder='public')

#paste model.py ignoring the libraries code and run once, so the model can be saved and it can be removed from here to decrease
#run time to get the result

model_path = 'my_project_model.h5'
model = load_model(model_path)
model.make_predict_function()
print("model loaded")  

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1,28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0

    result = model.predict(img)
    preds = np.argmax(result)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        res = model_predict(file_path, model)
        preds = str(res)
        return preds

if __name__ == '__main__':
    app.run(debug=True)
