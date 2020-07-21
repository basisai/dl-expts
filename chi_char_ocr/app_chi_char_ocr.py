import codecs

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras import backend as K

from .utils import model08

DATA_DIR = "chi_char_ocr/"
IMG_SIZE = 96


@st.cache
def preprocess(raw_img):
    img = np.asarray(raw_img.convert('L'))
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 3)
    img = img.astype(np.float32)
    img /= 255.0
    return img


@st.cache
def load_klasses():
    with codecs.open(DATA_DIR + "models/labels.txt", "r", "UTF-8") as label_file:
        klasses = [a.strip() for a in label_file.readlines()]
    return klasses


@st.cache(allow_output_mutation=True)
def load_model(n_classes):
    model = model08(IMG_SIZE, n_classes)
    model.load_weights(DATA_DIR + "models/weights08.h5")
    model._make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    session = K.get_session()
    return model, session


def rank_predictions(pred, klasses):
    top_preds = pd.DataFrame({"Character": klasses, "Probability": pred * 100})
    top_preds.sort_values("Probability", ascending=False, inplace=True, ignore_index=True)
    return top_preds


SAMPLES = [f"img{i}.png" for i in range(19)]


def chi_char_ocr():
    st.title("Handwritten Simplified Chinese Character Recognition Demo")

    klasses = load_klasses()
    model, session = load_model(len(klasses))

    # select_sample = st.selectbox(
    #     "Select a sample image or upload an image.",
    #     ["", "Upload an image"] + [f"ex{i + 1}" for i in range(len(SAMPLES))],
    # )

    # uploaded_file = None
    # if select_sample != "" and select_sample != "Upload an image":
    #     uploaded_file = "samples/" + SAMPLES[int(select_sample[2:]) - 1]
    # elif select_sample == "Upload an image":
    #     uploaded_file = st.file_uploader("Upload an image.")

    select = st.selectbox("", ["Select a sample image", "Upload an image"])

    if select == "Select a sample image":
        select_idx = st.slider("Select a sample image.", 0, len(SAMPLES))
        st.image(DATA_DIR + "samples/img_phrase.png", use_column_width=True)
        uploaded_file = None
        if select_idx > 0:
            uploaded_file = DATA_DIR + "samples/" + SAMPLES[int(select_idx) - 1]
    else:
        uploaded_file = st.file_uploader("Upload an image.")

    if uploaded_file is not None:
        st.subheader("Image")
        raw_img = Image.open(uploaded_file)
        st.image(raw_img, use_column_width=False)

        # Convert image to numpy.ndarray
        img = preprocess(raw_img)

        with session.as_default():
            pred = model.predict(img)[0]

        top_preds = rank_predictions(pred, klasses)

        c, p = top_preds.iloc[0].values
        st.subheader(f"Output: **`{c}`**")
        st.write("Top 10")
        st.write(top_preds.iloc[:10])


if __name__ == "__main__":
    chi_char_ocr()
