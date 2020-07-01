import pytesseract
import streamlit as st
from PIL import Image


@st.cache
def recognize(img):
    return pytesseract.image_to_string(img)


def image_ocr():
    st.title("Image OCR Demo")

    uploaded_file = st.file_uploader("Upload an image.")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        st.image(img, caption="Uploaded File", use_column_width=True)

        st.subheader("Output")
        output_text = recognize(img)
        st.text(output_text)
        

if __name__ == "__main__":
    image_ocr()
