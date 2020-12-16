version = "1.0"

train {
    step train {
        image = "tensorflow/tensorflow:2.4.0-gpu"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "2"
            memory = "4G"
        }
    }
}
