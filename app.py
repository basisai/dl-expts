import base64
from pathlib import Path

import streamlit as st

from bert.app_ner import ner
from pneumonia.app_pneumonia import image_recognize
from pose_estimation.app_pose import pose_compare
from sentiment.app_sentiment_demo import demo_sentiment_analyzer
from autoencoder.app_autoencoder import demo_anomaly_detection
from handwriting_ocr_chi.app_handwriting_ocr_chi import chi_char_ocr
from handwriting_ocr_eng.app_handwriting_ocr_eng import eng_ocr
# from image_ocr.app_ocr import image_ocr
# from table_ocr.app_ocr import table_ocr


def uri_encode_path(path, mime="image/png"):
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def add_header(path):
    st.markdown(
        "<img src='{}' class='img-fluid'>".format(uri_encode_path(path)),
        unsafe_allow_html=True,
    )


def main():
    # max_width = 900
    # st.markdown(
    #     f"""
    #     <style>
    #     .reportview-container .main .block-container{{
    #         max-width: {max_width}px;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    dict_pages = {
        "Named Entity Recognition": ner,
        "Chest X-ray Image Classification": image_recognize,
        "Pose Comparison": pose_compare,
        "Sentiment Analysis": demo_sentiment_analyzer,
        "Handwriting Recognition for English": eng_ocr,
        "Handwriting Recognition for Chinese": chi_char_ocr,
        "Anomaly Detection": demo_anomaly_detection,
        # "Table OCR": table_ocr,
        # "Image OCR": image_ocr,
    }

    select_page = st.sidebar.selectbox("Select a demo", [""] + list(dict_pages.keys()))

    if select_page == "":
        add_header("assets/logo.png")
        st.title("Demos")
        st.write("This app contains a series of demos. Select a demo in the left panel.")
        st.markdown("[Source code](https://github.com/basisai/dl-expts)")
    else:
        dict_pages[select_page]()

    st.sidebar.info(
        "**Note**: When querying Bedrock endpoints, for\n"
        "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
        "replace **http** with **https** in the API URL.")


if __name__ == "__main__":
    main()
