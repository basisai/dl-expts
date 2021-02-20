"""
Streamlit app
"""
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = "autoencoder/results/"


@st.cache
def read_data():
    """Load data."""
    test = pd.read_csv(DATA_DIR + "test.csv")
    preds = pd.read_csv(DATA_DIR + "preds.csv")
    return test.values, preds.values


def demo_anomaly_detection():
    """App."""
    st.title("Anomaly Detection Using Autoencoder Demo")
    st.subheader("Analysis Method")
    st.write(
        """
        - Using historical time series data, we train an autoencoder model to learn a
        representation of the data in an unsupervised manner.
        - At serving time, a time series is then fed into the model to predict the expected shape.
        - We then compute the MSE between the actual time series and its prediction.
        - If the MSE exceeds a threshold, we conclude that the time series is an anomaly.
        """
    )

    st.subheader("Examples")
    test, preds = read_data()
    select_ex = st.selectbox("Select examples.", [f"ex{i}" for i in range(1, test.shape[0] + 1)])

    idx = int(select_ex[2:]) - 1
    st.write(f"**MSE: {np.linalg.norm(test[idx] - preds[idx]):.4f}**")
    st.line_chart(pd.DataFrame({"actual": test[idx], "prediction": preds[idx]}))


if __name__ == "__main__":
    demo_anomaly_detection()
