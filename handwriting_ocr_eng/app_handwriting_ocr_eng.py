"""
Streamlit app
"""
import base64
import json
from io import BytesIO
import requests

import streamlit as st
from PIL import Image

from .ocr.utils.cer_wer import cer_scores, wer_scores

DATA_DIR = "handwriting_ocr_eng/samples/"


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
    """OCR."""
    encoded_img = encode_image(image)
    data = json.dumps({"encoded_image": encoded_img})

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    decoded_text = response.json()["decoded_text"]
    return decoded_text


def compare(reference, decoded_text):
    """Compare reference text with decoded text."""
    cer_s, cer_d, cer_i = cer_scores(reference, decoded_text)
    st.write(f"**Overall CER: `{100 * (cer_s + cer_d + cer_i):.2f}%`**")
    st.write(f"- Substitution error = `{100 * cer_s:.2f}%`\n"
             f"- Deletion error = `{100 * cer_d:.2f}%`\n"
             f"- Insertion error = `{100 * cer_i:.2f}%`\n")

    wer_s, wer_d, wer_i = wer_scores(reference, decoded_text)
    st.write(f"**Overall WER: `{100 * (wer_s + wer_d + wer_i):.2f}%`**")
    st.write(f"- Substitution error = `{100 * wer_s:.2f}%`\n"
             f"- Deletion error = `{100 * wer_d:.2f}%`\n"
             f"- Insertion error = `{100 * wer_i:.2f}%`\n")


@st.cache
def load_results():
    """Load data."""
    return json.load(open(DATA_DIR + "results.json", "r"))


def eng_ocr():
    """App."""
    st.title("Handwriting Recognition for English")

    select_mode = st.selectbox("Choose a mode.", ["Select a sample image", "Upload an image"])

    if select_mode == "Select a sample image":
        samples = {
            "ex1": {
                "raw_img": "ex1.png",
                "segmented_img": "segmented1.png",
            },
            "ex2": {
                "raw_img": "ex2.png",
                "segmented_img": "segmented2.png",
            },
            "ex3": {
                "raw_img": "ex3.png",
                "segmented_img": "segmented3.png",
            },
            "ex4": {
                "raw_img": "ex4.png",
                "segmented_img": "segmented4.png",
            },
        }
        results = load_results()

        select_ex = st.selectbox("Select a sample image.", [""] + list(samples.keys()))
        if select_ex != "":
            sample = samples[select_ex]
            sample.update(results[select_ex])

            raw_img = Image.open(DATA_DIR + sample["raw_img"])
            st.image(raw_img, caption="Sample Image", use_column_width=True)

            st.write("**Segmentation**")
            segmented_img = Image.open(DATA_DIR + sample["segmented_img"])
            st.image(segmented_img, use_column_width=True)

            st.write("**Handwriting recognition**")
            decoded_text_am = "\n".join(sample["intermediate"])
            st.text(decoded_text_am)

            st.subheader("Denoised Output")
            decoded_text = sample["prediction"]
            st.text(decoded_text)

            st.subheader("Reference")
            reference = sample["reference"]
            st.text(reference)
            compare(reference, decoded_text)

    elif select_mode == "Upload an image":
        url = st.text_input("Input API URL.")
        token = st.text_input("Input token.")
        uploaded_file = st.file_uploader("Upload an image.")

        if uploaded_file is not None and url != "":
            raw_img = Image.open(uploaded_file)
            st.image(raw_img, use_column_width=False)

            decoded_text = recognize(raw_img, url, token)

            st.subheader("Output")
            st.text(decoded_text)

            st.subheader("Reference")
            reference = st.text_area("Input reference to compute CER and WER.")
            if reference != "":
                st.text(reference)
                compare(reference, decoded_text)

            
if __name__ == "__main__":
    eng_ocr()
