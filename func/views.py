from flask import render_template, request
import os
from PIL import Image
from requests import Response
from func.utils import *

UPLOAD_FLODER = 'static/uploads'


def base():
    return render_template('base.html')


def index():
    return render_template('index.html')


def faceapp():
    return render_template('faceapp.html')


def face_realtime():
    return render_template('face_realtime.html')


def getwidth(path):
    img = Image.open(path)
    size = img.size  # width and height
    aspect = size[0] / size[1]  # width / height
    w = 300 * aspect
    return int(w)


def face_recog():
    if request.method == "POST":
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FLODER, filename)
        f.save(path)
        w = getwidth(path)
        pipeline_model(path, filename)
        return render_template('face_recog.html', fileupload=True, img_name=filename, w=w)

    return render_template('face_recog.html', fileupload=False, img_name="freeai.png")
