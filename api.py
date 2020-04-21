import base64
import cv2
import numpy as np
import flask
import io
from PIL import Image
from flask import jsonify

from low_light import low_image_enhancement
from detection import detect

app = flask.Flask(__name__)


@app.route("/lowlight", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            npimg = np.fromstring(image, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = low_image_enhancement(img)
            img = Image.fromarray(img.astype("uint8"))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
            data['success'] = True
            data['image'] = str(img_base64)
    return jsonify(data)


@app.route("/detect", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            npimg = np.fromstring(image, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            img = detect(img)
            img = Image.fromarray(img.astype("uint8"))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read())
            data['success'] = True
            data['image'] = str(img_base64)
    return jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    app.run(debug=True)