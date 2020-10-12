"""
Streamlit app
"""
import json
import requests

import streamlit as st


COLOUR_MAP = {
    "PERSON": "#00FEFE", # cyan
    "ORGANIZATION": "#FFFF00", # yellow
    "LOCATION": "#00FF00", # green
}

_highlighted = "<span style='background-color: {colour}'>{word}</span>"
SAMPLES = json.load(open("ner/samples.json", "r"))

    
def print_text_with_tags(text_split, tags):
    print_str = []
    for word, tag in zip(text_split, tags):
        c = COLOUR_MAP.get(tag)
        if c is not None:
            print_str.append(_highlighted.format(colour=c, word=word))
        else:
            print_str.append(word)
    return " ".join(print_str)


@st.cache
def predict(input_text, zh, url, token):
    data = {"text": input_text}
    if zh == "Yes":
        data["lang"] = "zh-cn"
    data = json.dumps(data)

    headers = {"Content-Type": "application/json"}
    if token != "":
        headers.update({"X-Bedrock-Api-Token": token})

    response = requests.post(url, headers=headers, data=data)
    text_split = response.json()["text_split"]
    tags = response.json()["tags"]
    return text_split, tags


def print_results(output):
    st.markdown(output, unsafe_allow_html=True)

    st.subheader("Legend")
    _print = []
    for k, v in COLOUR_MAP.items():
        _print.append(_highlighted.format(colour=v, word=k))
    st.markdown(" ".join(_print), unsafe_allow_html=True)

    
def ner():
    st.title("Named Entity Recognition Demo")

    select_mode = st.selectbox("Choose a mode.", ["Select a sample text", "Input text"])

    if select_mode == "Select a sample text":

        select_ex = st.selectbox("Select a sample text.", [""] + list(SAMPLES.keys()))
        if select_ex != "":
            sample = SAMPLES[select_ex]
            st.subheader("Input:")
            st.write(sample["text"])

            st.subheader("Output:")
            print_results(sample["output"])

    elif select_mode == "Input text":
        url = st.text_input("Input API URL.")
        token = st.text_input("Input token.")
        input_text = st.text_area("Input text.")
        zh = st.radio("Select 'Yes' if the text input is Chinese.", ["No", "Yes"])

        st.write("**Samples**")
        for _, v in SAMPLES.items():
            st.code(v["text"])

        if input_text != "" and url != "":
            st.subheader("Input:")
            st.write(input_text)

            text_split, tags = predict(input_text, zh, url, token)
            output_text = print_text_with_tags(text_split, tags)

            st.subheader("Output:")
            print_results(output_text)


if __name__ == "__main__":
    ner()
