import streamlit as st

from bert.app_ner import ner
from table_ocr.app_ocr import table_ocr
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
        "Table OCR",
        "Named Entity Recognition",
    ])

    if select_page == "Named Entity Recognition":
        ner()
    elif select_page == "Handwriting Recognition for English":
        st.write("*Under Construction*")
#         handwriting_ocr()
    elif select_page == "Table OCR":
        table_ocr()


if __name__ == "__main__":
    main()
