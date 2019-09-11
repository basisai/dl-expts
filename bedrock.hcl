version = "1.0"

train {
    image = "python:3.7"
    install = ["pip3 install -r requirements.txt"]
    script = ["python3 train.py"]

    parameters {
    }
}