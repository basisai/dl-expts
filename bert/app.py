import json
import requests

import streamlit as st

COLOUR_MAP = {
    "PERSON": "#00FEFE", # cyan
    "ORGANIZATION": "#FFFF00", # yellow
    "LOCATION": "#00FF00", # green
}

_highlighted = "<span style='background-color: {colour}'>{word}</span>"
    
    
def print_text_with_tags(text_split, tags):
    print_str = []
    for word, tag in zip(text_split, tags):
        c = COLOUR_MAP.get(tag)
        if c is not None:
            print_str.append(_highlighted.format(colour=c, word=word))
        else:
            print_str.append(word)
    return " ".join(print_str)

    
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
    
    input_text = st.text_input("Input text.")
    zh = st.radio("Select 'Yes' if the input text is Chinese.", ["No", "Yes"])
    
    if input_text != "":
        st.subheader("Input:")
        st.write(input_text)

        data = {"text": input_text}
        if zh == "Yes":
            data["lang"] = "zh-cn"
        data = json.dumps(data)

        url = "https://snowy-band-4038.pub.playground.bdrk.ai"
        headers = {"Content-Type": "application/json"}
        if token != "":
            headers.update({"X-Bedrock-Api-Token": token})
        
        response = requests.post(url, headers=headers, data=data)

        text_split = response.json()["text_split"]
        tags = response.json()["tags"]
        output_text = print_text_with_tags(text_split, tags)

        st.subheader("Output:")
        st.markdown(output_text, unsafe_allow_html=True)
        
    st.subheader("Legend")
    _print = []
    for k, v in COLOUR_MAP.items():
        _print.append(_highlighted.format(colour=v, word=k))
    st.markdown(" ".join(_print), unsafe_allow_html=True)

    st.header("Samples:")
    st.code("Mantan Gubernur DKI Jakarta Basuki Tjahaja Purnama atau Ahok menyatakan, dia heran Gubernur Anies "
            "Baswedan menerbitkan surat izin mendirikan bangunan (IMB) untuk bangunan di Pulau D, pulau hasil "
            "reklamasi, berdasarkan Peraturan Gubernur Nomor 206 Tahun 2016 yang dulu diteken Ahok.")

    st.code("腾讯科技股份有限公司是中國大陸规模最大的互联网公司，1998年11月由马化腾、张志东、陈一丹、许晨晔、曾李青5位创始人共同创立，"
            "總部位於深圳南山区騰訊濱海大廈。腾讯业务拓展至社交、娱乐、金融、资讯、工具和平台等不同领域。目前，腾讯拥有中国大陸使用人数"
            "最多的社交软件腾讯QQ和微信，以及最大的网络游戏社区腾讯游戏。在電子書領域 ，旗下有閱文集團，運營有QQ讀書和微信讀書。")

    st.code("Google, headquartered in Mountain View (1600 Amphitheatre Pkwy, Mountain View, CA 940430), "
            "unveiled the new Android phone for $799 at the Consumer Electronic Show. Sundar Pichai said in his "
            "keynote that users love their new Android phones.")


if __name__ == "__main__":
    main()
