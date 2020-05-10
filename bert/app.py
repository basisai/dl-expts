import json
import requests

import streamlit as st


def print_text_with_tags(text_split, tags):
    # 0 black
    # 1 red
    # 2 green
    # 3 yellow
    # 4 blue
    # 5 magenta
    # 6 cyan
    # 7 white
    # 9 default

    dict_background = {
        "PERSON": "\033[46m", # cyan
        "ORGANIZATION": "\033[43m", # yellow
        "LOCATION": "\033[45m" # magenta
    }
    for k, v in dict_background.items():
        print(v+k+"\033[49m")

    print_str = []
    for word, tag in zip(text_split, tags):
        c = dict_background.get(tag)
        if c is not None:
            print_str.append(c+word+"\033[49m")
        else:
            print_str.append(word)
        
    print(" ".join(print_str))

    
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
    
    st.title("Named Entity Recognition")
    
    token = st.text_input("Input token.")

    zh = st.radio("Chinese?", ["No", "Yes"])
    text = st.text_input("Input text.")
    
    if text != "":
        data = {"text": text}
        if zh == "Yes":
            data["lang"] = "zh-cn"
        data = json.dumps(data)

        url = "https://morning-frog-4457.pub.playground.bdrk.ai"
        headers = {"Content-Type": "application/json"}
        if token != "":
            headers.update({"X-Bedrock-Api-Token": token})
        
        response = requests.post(url, headers=headers, data=data)
        
        text_split = response.json()["text_split"]
        tags = response.json()["tags"]
        st.text(print_text_with_tags(text_split, tags))
        

if __name__ == "__main__":
    main()
