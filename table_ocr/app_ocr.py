import os
import tempfile

import pdf2image
import streamlit as st
import tabula
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


@st.cache
def convert_to_images(uploaded_file):
    return pdf2image.convert_from_bytes(uploaded_file.read())


@st.cache
def recognize(uploaded_file):
    pdf_file = uploaded_file.read()
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(pdf_file)
            f.flush()
            return tabula.read_pdf(f.name, pages="all")
    finally:
        os.close(fh)
        os.remove(temp_filename)


def table_ocr():
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

    st.title("Table OCR Demo")

    uploaded_file = st.file_uploader("Upload a PDF.")
    if uploaded_file is not None:
        images = convert_to_images(uploaded_file)
        dfs = recognize(uploaded_file)

        page_num = 0
        if len(images) > 1:
            page_num = st.slider("Select page", 1, len(images), 1) - 1

        st.subheader("Uploaded file")
        st.image(images[page_num], use_column_width=True)

        st.subheader("Output")
        st.dataframe(dfs[page_num])


if __name__ == "__main__":
    table_ocr()
