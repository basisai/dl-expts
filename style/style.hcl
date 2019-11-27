version = "1.0"

train {
    image = "tensorflow/tensorflow:2.0.0-gpu-py3"
    install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
    script = [{sh = ["python3 train_style.py"]}]

    parameters {
    }
}
