version = "1.0"

train {
    image = "tensorflow/tensorflow:1.14.0-gpu-py3"
    install = ["pip3 install -r requirements_tensorflow.txt"]
    script = [
        {
            sh = ["python3 train_dcgan.py"]
        }
    ]

    parameters {
    }
}
