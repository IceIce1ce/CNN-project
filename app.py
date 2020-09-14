import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

STATIC_FOLDER = 'static' 
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads' # folder save image uploaded to predict
MODEL_FOLDER = STATIC_FOLDER + '/models' # folder save our model trained

#########use for model 1 values in np array##########
'''
def load__model():
    global model
    model = load_model(MODEL_FOLDER + '/cat_dog_classifier.h5') # load model from folder models
    global graph
    #graph = tf.get_default_graph() # just use this with old version of tensorflow

def predict(fullpath):
    data = image.load_img(fullpath, target_size=(128, 128, 3)) # load image
    data = np.expand_dims(data, axis=0)
    data = data.astype('float') / 255 # scale by 255
    result = model.predict(data) # predict result
    # just use this with old version of tensorflow
    #with graph.as_default():
        #result = model.predict(data)
    return result
'''
#########use for model 1 values in np array##########

#########use for model 2 values in np array########
'''
def load__model():
    global model
    model = load_model(MODEL_FOLDER + '/my_model.h5')
    global graph
    #graph = tf.get_default_graph()
def predict(fullpath):
    data = image.load_img(fullpath, target_size=(128, 128, 3)) # load image
    data = np.expand_dims(data, axis=0)
    data = data.astype('float') / 255 # scale by 255
    result = model.predict(data) # predict result
    # just use this with old version of tensorflow
    #with graph.as_default():
        #result = model.predict(data)
    return result
'''
#########use for model 2 values in np array########

########use for resnet#######
def load__model():
    global model 
    model = load_model(MODEL_FOLDER + '/resnet_best.h5')
    global graph
    #graph = tf.get_default_graph() 
def predict(fullpath):
    data = cv2.imread(fullpath)
    data1 = cv2.resize(data, (224, 224))
    data1 = np.reshape(data1, [1, 224, 224, 3])
    result = model.predict(data1)
    # just use this with old version of tensorflow
    #with graph.as_default():
        #result = model.predict(data1)
    return result
########use for resnet#######

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image'] 
        fullname = os.path.join(UPLOAD_FOLDER, file.filename) # get image file from folder upload
        file.save(fullname) # save image for predicting
        '''
        result = predict(fullname) # predict image
        pred_prob = result.item() # get % accuracy prediction
        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)
        '''
        ##########use for resnet##########
        result = predict(fullname)
        x = result[0,0] # result of label cat
        y = result[0,1] # result of label dog
        if x > y:
            label = 'Cat'
            accuracy = round(x * 100, 2)
        else:
            label = 'Dog'
            accuracy = round(y * 100, 2)
        ##########use for resnet##########
        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def create_app():
    load__model()
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)