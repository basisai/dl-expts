import streamlit as st

from bert.app_ner import ner
# from handwriting_ocr_eng.app_ocr import handwriting_ocr

    
def main():
    # max_width = st.sidebar.slider("Set page width", 700, 1500, 1000, 20)
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
    
    select_page = st.sidebar.selectbox("Select page", [
        "Named Entity Recognition",
        "Handwriting Recognition for English",
    ])
    
    if select_page == "Named Entity Recognition":
        ner()
    elif select_page == "Handwriting Recognition for English":
        st.write("*Under Construction*")
#         handwriting_ocr()


if __name__ == "__main__":
    main()
