import json
import requests

import streamlit as st
from PIL import Image

from .utils_image import encode_image, decode_image

DATA_DIR = "pneumonia/"
SAMPLES = {
    "ex1": "covid-19-pneumonia-67.jpeg",
    "ex2": "covid-19-caso-82-1-8.png",
    "ex3": "41182_2020_203_Fig4_HTML.jpg",
    "ex4": "pneumococcal-pneumonia-day0.jpg",
}


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    return response.json()


def image_recognize():
    st.title("Chest X-ray Image Classification Demo")

    url = st.text_input("Input API URL.", "https://shiny-mouse-9036.pub.playground.bdrk.ai")
    token = st.text_input("Input token.")

    select_mode = st.selectbox("Choose a mode.", ["", "Select a sample image", "Upload an image"])

    uploaded_file = None
    if select_mode == "Select a sample image":
        select_eg = st.selectbox("Select a sample image.", list(SAMPLES.keys()))
        uploaded_file = DATA_DIR + "samples/" + SAMPLES[select_eg]
    elif select_mode == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image.")

    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        response_json = recognize(image, url, token)
        prob = response_json["prob"] * 100
        cam_image = decode_image(response_json["cam_image"])
        gc_image = decode_image(response_json["gc_image"])

        st.subheader(f"Probability of having COVID-19 = `{prob:.2f}%`")
        st.header("Explainability")
        st.subheader("[Grad-CAM and Guided Grad-CAM](http://gradcam.cloudcv.org/)")
        st.write("To visualise the regions of input that are 'important' for predictions from "
                 "Convolutional Neural Network (CNN)-based models.")
        st.image(cam_image, caption="Grad-CAM Image", width=300)
        st.image(gc_image, caption="Guided Grad-CAM Image", width=300)


if __name__ == "__main__":
    image_recognize()
