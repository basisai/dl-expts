import base64
import json
import requests
from io import BytesIO

import streamlit as st
from PIL import Image

from .ocr.utils.cer_wer import cer_scores, wer_scores

DATA_DIR = "handwriting_ocr_eng/"


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
    decoded_text = response.json()["decoded_text"]
    return decoded_text


def compare(reference, decoded_text):
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
    with open(DATA_DIR + "samples/results.json", "r") as f:
        results = json.load(f)
    return results


def eng_ocr():
    st.title("Handwriting Recognition for English")

    results = load_results()
    select = st.selectbox("", ["Select a sample image", "Upload an image"])

    if select == "Select a sample image":
        select_ex = st.selectbox("Select a sample image.", [""] + [f"ex{i}" for i in range(1, 5)])
        if select_ex != "":
            raw_img = Image.open(DATA_DIR + f"samples/{select_ex}.png")
            st.image(raw_img, caption="Sample Image", use_column_width=True)

            st.write("**Segmentation**")
            segmented_img = Image.open(DATA_DIR + f"samples/segmented{int(select_ex[2:])}.png")
            st.image(segmented_img, use_column_width=True)

            st.write("**Handwriting recognition**")
            decoded_text_am = "\n".join(results[select_ex]["intermediate"])
            st.text(decoded_text_am)

            st.subheader("Denoised Output")
            decoded_text = results[select_ex]["prediction"]
            st.text(decoded_text)

            st.subheader("Reference")
            reference = results[select_ex]["reference"]
            st.text(reference)
            compare(reference, decoded_text)

    else:
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
