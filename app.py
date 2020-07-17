import base64
from pathlib import Path

import streamlit as st

from bert.app_ner import ner
from pneumonia.app_pneumonia import image_recognize
from image_ocr.app_ocr import image_ocr
from table_ocr.app_ocr import table_ocr
from pose_estimation.app_pose import pose_compare
from sentiment.app_sentiment_demo import demo_sentiment_analyzer
# from handwriting_ocr_eng.app_ocr import handwriting_ocr


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

    select_page = st.sidebar.selectbox("Select demo", [
        "",
        "Named Entity Recognition",
        "Chest X-ray Image Classification",
        "Pose Comparison",
        "Sentiment Analysis",
        "Image OCR",
        "Table OCR",
    ])

    if select_page == "":
        add_header("assets/logo.png")
        st.title("Demos")
        st.write("This app contains a series of demos. Select a demo in the left panel.")
    elif select_page == "Named Entity Recognition":
        ner()
    elif select_page == "Chest X-ray Image Classification":
        image_recognize()
    elif select_page == "Pose Comparison":
        pose_compare()
    elif select_page == "Sentiment Analysis":
        demo_sentiment_analyzer()
    elif select_page == "Table OCR":
        table_ocr()
    elif select_page == "Image OCR":
        image_ocr()
    elif select_page == "Handwriting Recognition for English":
        st.write("*Under Construction*")
#         handwriting_ocr()


if __name__ == "__main__":
    main()
