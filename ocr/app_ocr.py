import pdf2image
import pytesseract
import streamlit as st
from PIL import Image
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


@st.cache
def recognize(img):
    return pytesseract.image_to_string(img)


def main():
    max_width = 900 #st.sidebar.slider("Set page width", min_value=700, max_value=1500, value=1000, step=20)
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("OCR Demo")
    
    select_type = st.radio("Select file type.", ["image", "PDF"])
    
    images = None
    if select_type == "image":
        uploaded_file = st.file_uploader("Upload an image.")
        if uploaded_file is not None:
            images = [Image.open(uploaded_file)]
    else:
        uploaded_file = st.file_uploader("Upload a PDF.")
        if uploaded_file is not None:
            images = pdf2image.convert_from_bytes(uploaded_file.read())
    
    if images:
        if len(images) > 1:
            page_num = st.slider("Select page", 1, len(images), 1) - 1
        else:
            page_num = 0

        st.image(images[page_num], caption="Uploaded File", use_column_width=True)

        st.subheader("Output")
        output_text = recognize(images[page_num])
        st.text(output_text)
        

if __name__ == "__main__":
    main()
