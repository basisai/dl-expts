"""
Script containing autoencoder utils.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import timedelta


def ts_plots(x_test, preds, scores, threshold=0, output_path=None):
    """Plot time series."""
    num = min(len(x_test), 30)  # max 30 plots
    sqrtn = int(np.ceil(np.sqrt(num)))

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    for i in range(num):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.plot(x_test[i])
        if scores[i] > threshold:
            ax.plot(preds[i], color="red")
        else:
            ax.plot(preds[i])
        ax.set_title("mse = {:.6f}".format(scores[i]))

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)


def loss_plot(history, output_path=None):
    """Plot losses."""
    fig = plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)


def compute_mse(model, x_data):
    """Compute mse between actual time series and prediction."""
    preds = np.squeeze(model.predict(np.expand_dims(x_data, axis=2)))
    scores = np.sum((x_data - preds) ** 2, axis=1) / x_data.shape[1]
    return scores, preds


def compute_threshold(model, x_train, outlier_prop):
    """Compute threshold."""
    scores, _ = compute_mse(model, x_train)
    scores.sort()
    return scores[int((1 - outlier_prop) * len(scores))]


def load_data(input_path, periods_per_day):
    """Load data."""
    raw_df = pd.read_csv(input_path)
    raw_df = raw_df.groupby("timestamp").mean().reset_index()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    start = raw_df["timestamp"].min().date()
    end = raw_df["timestamp"].max().date() + timedelta(days=1)
    num_days = (end - start).days
    data = pd.DataFrame({"timestamp": pd.date_range(
        start, end, periods=num_days * periods_per_day + 1)}).iloc[:-1]
    data = pd.merge(data, raw_df, on="timestamp", how="left")
    data.set_index("timestamp", inplace=True)
    data.fillna(method="ffill", inplace=True)
    print("Data shape:", data.shape)
    return data


def train_val_split(data, periods_per_day, train_size=0.8):
    """Split data to train and val."""
    split = int(len(data) / periods_per_day * train_size) * periods_per_day
    return data.values[:split], data.values[split:]
