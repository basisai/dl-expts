import base64
import json
import requests
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

DATA_DIR = "handwriting_ocr_chi/"
IMG_SIZE = 96
SAMPLES = [f"img{i}.png" for i in range(19)]


def encode_image(image):
    """Encode an image to base64 encoded bytes.
    Args:
        image: PIL.PngImagePlugin.PngImageFile
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


@st.cache
def recognize(image, url, token):
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    character = response.json()["character"]
    prob = response.json()["prob"]
    return character, prob


@st.cache
def load_results():
    return pd.read_csv(DATA_DIR + "results.csv")


def chi_char_ocr():
    st.title("Handwriting Recognition Demo for Simplified Chinese Character")

    results = load_results()

    select = st.selectbox("", ["Select a sample image", "Upload an image"])

    if select == "Select a sample image":
        select_idx = st.slider("Select a sample image.", 0, len(SAMPLES))
        st.image(DATA_DIR + "samples/img_phrase.png", use_column_width=True)
        uploaded_file = None
        if select_idx > 0:
            uploaded_file = DATA_DIR + "samples/" + SAMPLES[int(select_idx) - 1]

        if uploaded_file is not None:
            st.subheader("Image")
            raw_img = Image.open(uploaded_file)
            st.image(raw_img, use_column_width=False)

            top_preds = results.query(f"ex == 'ex{int(select_idx) - 1}'")[["Character", "Probability"]]
            character = top_preds["Character"].iloc[0]
            st.subheader(f"Output: **`{character}`**")
            st.write("Top 10")
            st.write(top_preds.iloc[:10])

    else:
        url = st.text_input("Input API URL.")
        token = st.text_input("Input token.")
        uploaded_file = st.file_uploader("Upload an image.")

        if uploaded_file is not None and url != "":
            raw_img = Image.open(uploaded_file)
            st.image(raw_img, use_column_width=False)

            character, prob = recognize(raw_img, url, token)
            st.subheader(f"Output: **`{character} ({100 * prob:.2f}%)`**")


if __name__ == "__main__":
    chi_char_ocr()
