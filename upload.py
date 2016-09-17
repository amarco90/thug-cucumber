#!/usr/bin/env python
# -*- coding: utf-8 -*-
import base64
import errno
import json
import os

from flask import (Flask, jsonify, redirect, request, send_from_directory,
                   url_for)

from backend.vision_api import VisionApi
from backend.preprocessing import Preprocessing
from flask_cors import CORS, cross_origin
from flask_sslify import SSLify

UPLOAD_FOLDER = '/tmp/uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        img = request.form['img']
        print img
        img = Preprocessing.preprocess(img)

        vision = VisionApi()
        print("Received Image of Size %d" % len(img))
        response = vision.detect_text(base64_str=img)
        print response
        return response

    return app.send_static_file('index.html')


@app.route('/base64/<filename>')
def get_base64(filename):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image:
        encoded_string = base64.b64encode(image.read())
        return encoded_string


@app.route('/ml_json/')
def ml_json():
    with open('./static/mock_data.json') as mock_data:
        return jsonify(json.load(mock_data))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    mkdir_p(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', debug=True)
