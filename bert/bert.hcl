version = "1.0"

train {
    step train {
        image = "tensorflow/tensorflow:2.1.0-gpu-py3"
        install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
        script = [{sh = ["python3 train_bertner.py"]}]
        resources {
            cpu = "2"
            memory = "4G"
        }
    }
}

serve{
    image = "python:3.7"
    install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
    script = [{sh = ["python3 serve_bertner.py"]}]
    resources {
        cpu = "2"
        memory = "4G"
    }
}
}
