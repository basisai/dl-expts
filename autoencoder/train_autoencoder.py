"""
Script to train autoencoder.
"""
import logging
import os
import pickle
import time

import numpy as np
from bedrock_client.bedrock.api import BedrockApi
from keras.models import Model, load_model
from keras.layers import (
    Input, Dense, Conv1D, GlobalMaxPool1D, RepeatVector, LSTM, Activation, TimeDistributed)
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler

from utils import (
    compute_mse, compute_threshold, load_data, loss_plot, ts_plots, train_val_split)

INPUT_TRAIN_FILE_PATH = os.getenv("INPUT_TRAIN_FILE_PATH")
INPUT_TEST_FILE_PATH = os.getenv("INPUT_TEST_FILE_PATH")
MODEL_TYPE = os.getenv("MODEL_TYPE")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS"))
OUTLIER_PROP = float(os.getenv("OUTLIER_PROP"))

PERIODS_PER_DAY = 96
SCALER_FILE_PATH = "/artefact/scaler.pkl"
OUTPUT_MODEL_PATH = f"/artefact/{MODEL_TYPE}_model.h5"
OUTPUT_IMG_PREFIX = "/artefact/plots/"


def build_autoencoder_cnn(timesteps, input_dim=1):
    """Build CNN autoencoder."""
    inputs = Input(shape=(timesteps, input_dim))

    # Encoder
    encoded = Conv1D(256, kernel_size=5, padding="same", activation="relu")(inputs)
    encoded = GlobalMaxPool1D()(encoded)
    # Decoder
    decoded = Dense(timesteps, activation="linear")(encoded)
    return Model(inputs, decoded)


def build_autoencoder_lstm(timesteps, input_dim=1):
    """Build LSTM autoencoder."""
    inputs = Input(shape=(timesteps, input_dim))

    # Encoder
    encoded = LSTM(32, return_sequences=False)(inputs)
    # Decoder
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(16, return_sequences=True)(decoded)
    decoded = LSTM(32, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)
    decoded = Activation("linear")(decoded)
    return Model(inputs, decoded)


def train(x_train, x_val):
    """Train model."""
    if MODEL_TYPE == "cnn":
        model = build_autoencoder_cnn(x_train.shape[1])
        y = x_train
        validation_data = (np.expand_dims(x_val, axis=2), x_val)
    elif MODEL_TYPE == "lstm":
        model = build_autoencoder_lstm(x_train.shape[1])
        y = np.expand_dims(x_train, axis=2)
        validation_data = (np.expand_dims(x_val, axis=2), np.expand_dims(x_val, axis=2))
    else:
        raise Exception("MODEL TYPE not found")
    print(model.summary())

    model.compile(optimizer="adam", loss="mse")

    checkpointer = ModelCheckpoint(
        filepath=OUTPUT_MODEL_PATH, verbose=0, save_best_only=True)
    history = model.fit(
        x=np.expand_dims(x_train, axis=2),
        y=y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        validation_data=validation_data,
        callbacks=[checkpointer],
    ).history
    return history


def main():
    """Train pipeline"""
    # from tensorflow.python.client import device_lib
    # print("List local devices")
    # print(device_lib.list_local_devices())
    #
    # print("\nGet available GPUs")
    # print(K.tensorflow_backend._get_available_gpus())

    os.mkdir(OUTPUT_IMG_PREFIX)

    print("\nLoading data")
    train_data = load_data(INPUT_TRAIN_FILE_PATH, PERIODS_PER_DAY)
    x_train, x_val = train_val_split(train_data, PERIODS_PER_DAY)
    x_test = load_data(INPUT_TEST_FILE_PATH, PERIODS_PER_DAY)

    # Scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    with open(SCALER_FILE_PATH, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    # Reshape
    x_train = x_train.reshape(-1, PERIODS_PER_DAY)
    x_val = x_val.reshape(-1, PERIODS_PER_DAY)
    x_test = x_test.reshape(-1, PERIODS_PER_DAY)

    print("\nTraining")
    start = time.time()
    history = train(x_train, x_val)
    print("\tTime taken = {:.2f} mins".format((time.time() - start) / 60))

    loss_plot(history, output_path=OUTPUT_IMG_PREFIX + "loss_plots.png")

    print("\nEvaluation")
    model = load_model(OUTPUT_MODEL_PATH)
    threshold = compute_threshold(model, x_train, OUTLIER_PROP)
    scores, preds = compute_mse(model, x_test)
    ts_plots(x_test, preds, scores, threshold=threshold,
             output_path=OUTPUT_IMG_PREFIX + "test_plots.png")

    print("\tThreshold = {:.6f}".format(threshold))
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Threshold", threshold)


if __name__ == "__main__":
    main()
