import streamlit as st

from bert.app_ner import ner
from image_ocr.app_ocr import image_ocr
from table_ocr.app_ocr import table_ocr
from pose_estimation.app_pose import pose_compare
# from handwriting_ocr_eng.app_ocr import handwriting_ocr

    
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
        "Named Entity Recognition",
        "Image OCR",
        "Table OCR",
        "Pose Comparison",
    ])

    if select_page == "Named Entity Recognition":
        ner()
    elif select_page == "Handwriting Recognition for English":
        st.write("*Under Construction*")
#         handwriting_ocr()
    elif select_page == "Table OCR":
        table_ocr()
    elif select_page == "Image OCR":
        image_ocr()
    elif select_page == "Pose Comparison":
        pose_compare()


if __name__ == "__main__":
    main()
