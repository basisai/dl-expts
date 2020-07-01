import os
import tempfile

import streamlit as st
import tabula


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
    st.title("Table OCR Demo")

    uploaded_file = st.file_uploader("Upload a PDF.")
    if uploaded_file is not None:
        dfs = recognize(uploaded_file)

        page_num = 0
        if len(dfs) > 1:
            page_num = st.slider("Select page", 1, len(dfs), 1) - 1

        st.subheader("Output")
        st.dataframe(dfs[page_num])


if __name__ == "__main__":
    table_ocr()
