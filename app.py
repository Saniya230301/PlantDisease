from gzip import FNAME
from flask import Flask,render_template,request,session
from werkzeug.utils import secure_filename
import os
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)

@app.route('/plntds', methods=['GET', 'POST'])
def plntds():
    print('hi')
    op = "abc"
   
    if request.method == "POST":
        # Get the file from post request
        op = "mcs"

        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/upload', secure_filename(f.filename))
        f.save(file_path)
        fname = secure_filename(f.filename)
        print("printing file name")
        print(fname)
    
        
        predict_dir_path = r'static/upload/'
        onlyfiles = [f for f in listdir(
            predict_dir_path) if isfile(join(predict_dir_path, f))]
        print(onlyfiles)

        
        model = tf.keras.models.load_model("plant_new_23_11_2022.hp5")
        image_size = 224
        #for file in onlyfiles:
        # img = keras.utils.load_img(predict_dir_path+file, target_size=(224, 224))
        # x = keras.utils.img_to_array(img)
        img = keras.utils.load_img(predict_dir_path+str(fname),
                                target_size=(image_size, image_size))
        x = keras.utils.img_to_array(img)
        print("printing X#######")
        print(x)
        x = x/255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = np.argmax(model.predict(images), axis=1)
        #classes = model.predict(images)
        print("printing classes")
        print(classes)
        
        list2 = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']
        print(classes[0])
        print(list2[classes[0]])
        
        op = str(list2[classes[0]])
    
    

        return render_template('index.html', op=op)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(port=3000, debug=True)
