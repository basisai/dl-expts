version = "1.0"

train {
    image = "python:3.7"
    install = ["pip3 install --upgrade pip && pip3 install -r requirements.txt"]
    script = [{sh = ["python3 train_autoencoder.py"]}]

    parameters {
        INPUT_TRAIN_FILE_PATH = "gs://bedrock-sample/at-load-actual-entsoe/train.csv"
        INPUT_TEST_FILE_PATH = "gs://bedrock-sample/at-load-actual-entsoe/test.csv"
        MODEL_TYPE = "cnn"
        BATCH_SIZE = "32"
        EPOCHS = "50"
        OUTLIER_PROP = "0.1"
    }
}

batch_score {
    image = "python:3.7"
    install = [
        "pip3 install --upgrade pip && pip3 install -r requirements.txt",
        "pip3 install neptune-client==0.4.92 Pillow==6.1.0 psutil==5.6.3"
    ]
    script = [{sh = ["python3 batch_score_autoencoder.py"]}]

    secrets = [
        "NEPTUNE_API_TOKEN",
        "NEPTUNE_PROJECT"
    ]

    parameters {
        INPUT_TEST_FILE_PATH = "gs://bedrock-sample/at-load-actual-entsoe/test.csv"
        MODEL_TYPE = "cnn"
    }
}
