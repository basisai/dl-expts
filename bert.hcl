version = "1.0"

train {
    image = "tensorflow/tensorflow:1.14.0-gpu-py3"
    install = ["pip3 install -U pip && pip install -r requirements_pytorch.txt"]
    script = [
        {
            sh = ["python3 train_bertner.py"]
        }
    ]

    parameters {
    }
}

serve{
    image = "python:3.7"
    install = ["pip3 install -r requirements_pytorch.txt && pip3 install jieba==0.39"]
    script = ["python3 serve_bertner.py"]
}
