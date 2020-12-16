version = "1.0"

// dummy in order to create pipeline in bedrock
train {
    step step0 {
        image = "python:3.6-slim"
        script = [
            {sh = ["python3 train.py"]}
        ]

        resources {
            cpu = "0.5"
            memory = "200M"
        }
    }
}

serve{
    image = "python:3.7"
    install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
    script = [
        {sh = ["streamlit run app_handwriting_ocr_eng_local.py --server.headless true --server.enableCORS=false --server.port ${BEDROCK_SERVER_PORT}"]}
    ]
}
