"""
Script for serving.
"""
import base64
import codecs
import copy

import numpy as np
import six
from keras import backend as K
from flask import Flask, request

from .utils import model08


IMG_SIZE = 96

with codecs.open("/artefact/labels.txt", "r", "UTF-8") as label_file:
    klasses = [a.strip() for a in label_file.readlines()]

model = model08(IMG_SIZE, len(klasses))
model.load_weights("/artefact/weights08.h5")
model._make_predict_function()
model.summary()  # included to make it visible when model is reloaded
session = K.get_session()


def decode_image(field, dtype=np.uint8):
    """Decode a base64 encoded numpy array to a list of floats.
    Args:
        field: base64 encoded string or bytes
        dtype
    Returns:
        numpy.array
    """
    if field is None:
        return None
    if not isinstance(field, bytes):
        field = six.b(field)
    array = np.frombuffer(base64.b64decode(field), dtype=dtype)
    return array


def preprocess(raw_img):
    img = np.asarray(raw_img.convert('L'))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    img = img.astype(np.float32)
    img /= 255.0
    return img


def top_predictions(pred, n=1):
    tops = []
    pred_copy = copy.copy(pred)
    for _ in range(n):
        i = np.argmax(pred_copy).item()
        tops.append([klasses[i], pred_copy[i]])
        pred_copy[i] = 0
    return tops


def predict(request_json):
    """Predict function."""
    raw_img = decode_image(request_json["encoded_image"]).reshape(
        request_json["image_shape"])
    img = preprocess(raw_img)

    with session.as_default():
        pred = model.predict(img)[0]

    top_pred = top_predictions(pred, n=1)[0]
    return top_pred


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    top_pred = predict(request.json)
    return {"character": top_pred[0], "prob": top_pred[1]}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
