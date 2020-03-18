version = "1.0"

serve{
    image = "python:3.7"
    install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
    script = [
        {sh = [
             "streamlit run app_eng.py --server.headless true --server.enableCORS=false --server.port ${BEDROCK_SERVER_PORT}",
        ]}
    ]

    parameters {
        WORKERS = "1"
    }
}
