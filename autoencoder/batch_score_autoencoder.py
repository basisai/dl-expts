"""
Script for performing batch scoring.
"""
import os
import pickle

import numpy as np
import neptune
from keras.models import load_model
from PIL import Image

from utils import compute_mse, load_data, ts_plots

NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
INPUT_TEST_FILE_PATH = os.getenv("INPUT_TEST_FILE_PATH")
MODEL_TYPE = os.getenv("MODEL_TYPE")

PERIODS_PER_DAY = 96
SCALER_FILE_PATH = "/artefact/scaler.pkl"
OUTPUT_MODEL_PATH = f"/artefact/{MODEL_TYPE}_model.h5"


def fig2pil(fig):
    """Convert figure to PIL image."""
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def main():
    """Entry point for batch scoring."""
    model = load_model(OUTPUT_MODEL_PATH)
    x_test = load_data(INPUT_TEST_FILE_PATH, PERIODS_PER_DAY)

    scaler = pickle.load(open(SCALER_FILE_PATH, "rb"))
    x_test = scaler.transform(x_test)
    x_test = x_test.reshape(-1, PERIODS_PER_DAY)

    scores, preds = compute_mse(model, x_test)

    neptune.init(
        api_token=NEPTUNE_API_TOKEN,
        project_qualified_name=NEPTUNE_PROJECT,
    )

    with neptune.create_experiment(
            name="autoencoders",
            params={"model_type": MODEL_TYPE, "test_data": INPUT_TEST_FILE_PATH}
        ):
        neptune.send_image("test_plots", fig2pil(ts_plots(x_test, preds, scores)))


if __name__ == "__main__":
    main()
