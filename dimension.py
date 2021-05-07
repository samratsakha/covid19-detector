from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from skimage.color import gray2rgb

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer


# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    img = Image.open(img_path)
    newsize = (224,224)
    img = img.resize(newsize)

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    X = gray2rgb(x)
    X = np.squeeze(X, axis=3)

    preds = model.predict(X)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Covid19 - Negative"
    elif preds==1:
        preds="Covid19 - Positive"
    
    
    return preds




f = 'image.jpeg'

# Save the file to ./uploads
"""
basepath = os.path.dirname(__file__)
file_path = os.path.join(
    basepath, 'uploads', secure_filename(f.filename))
f.save(file_path)
"""


# Make prediction
preds = model_predict(f, model)
result=preds

print(result)


