"""
Streamlit app entry page
"""
import streamlit as st

from ner.app_ner import ner
from pneumonia.app_pneumonia import image_recognize
from detection.app_detection import detect
from pose_estimation.app_pose import pose_compare
from sentiment.app_sentiment_demo import demo_sentiment_analyzer
from autoencoder.app_autoencoder import demo_anomaly_detection
from handwriting_ocr_chi.app_handwriting_ocr_chi import chi_char_ocr
from handwriting_ocr_eng.app_handwriting_ocr_eng import eng_ocr
# from image_ocr.app_ocr import image_ocr
# from table_ocr.app_ocr import table_ocr


def main():
    """App."""
    dict_pages = {
        "Named Entity Recognition": ner,
        "X-ray Image Classification": image_recognize,
        "Object Detection and Tracking": detect,
        "Pose Comparison": pose_compare,
        "Sentiment Analysis": demo_sentiment_analyzer,
        "Handwriting Recognition (English)": eng_ocr,
        "Handwriting Recognition (Chinese)": chi_char_ocr,
        "Anomaly Detection": demo_anomaly_detection,
        # "Table OCR": table_ocr,
        # "Image OCR": image_ocr,
    }

    st.sidebar.title("Demos")
    select_page = st.sidebar.radio("Go to", ["Home"] + list(dict_pages.keys()))

    if select_page == "Home":
        st.image("./assets/logo.png")
        st.title("Demos")
        st.write("This app contains a series of demos. Select a demo on the left panel.")
        st.markdown("[Source code](https://github.com/basisai/dl-expts)")
    else:
        dict_pages[select_page]()

    # st.sidebar.info(
    #     "**Note**: When querying Bedrock endpoints, for\n"
    #     "> `ConnectionError: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))`\n\n"
    #     "replace **http** with **https** in the API URL.")


if __name__ == "__main__":
    main()
