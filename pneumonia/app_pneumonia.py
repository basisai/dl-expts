import base64
import json
import requests

import numpy as np
import streamlit as st
from PIL import Image


def encode_image(array, dtype=np.uint8):
    """Encode an array to base64 encoded string or bytes.
    Args:
        array: numpy.array
        dtype
    Returns:
        base64 encoded string
    """
    if array is None:
        return None
    return base64.b64encode(np.asarray(array, dtype=dtype)).decode("utf-8")


@st.cache
def recognize(image, url, token):
    img = np.asarray(image.convert("RGB"))
    encoded_img = encode_image(img.ravel())
    data = json.dumps({"encoded_image": encoded_img, "image_shape": img.shape})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    prob = response.json()["prob"]
    return prob


def image_recognize():
    st.title("Chest X-ray Image Classification Demo")
    
    url = st.text_input("Input API URL.")
    token = st.text_input("Input token.")

    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None and url != "":
        image = Image.open(uploaded_file)

        prob = recognize(image, url, token)
        st.subheader(f"Probability of having COVID-19 = {prob:.6f}")
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        

if __name__ == "__main__":
    image_recognize()
