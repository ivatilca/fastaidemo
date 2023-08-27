import os
import urllib
from io import BytesIO
#from PIL import Image
from flask import Flask,request,redirect,url_for,render_template,flash
from werkzeug.utils import secure_filename

import json, requests
import os, base64
import urllib
#from PIL import Image

scoring_uri = 'http://7724f32c-4659-4a04-823d-39ef71e678c5.westus.azurecontainer.io/score'
headers = {'Content-Type': 'application/json'}
def preprocess(image):
    # Image = requests.get(url)
    # with open("test.jpg","wb") as f:
    #   f.write(Image.content)
    with open(image , mode='rb') as file:
        test = file.read()

    data = str(base64.b64encode(test), encoding='utf-8')  
    input_data = json.dumps({'data': data})
    return input_data


# create the application project 
app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(str(file_path))

        # Make prediction
        input_data = preprocess(str(file_path))
        preds = requests.post(scoring_uri,input_data,headers=headers)
        result = preds.text
        return result
    return None

# if __name__=='__main__':
#     app.run(debug=True)