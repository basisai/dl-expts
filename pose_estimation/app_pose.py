"""
Streamlit app
"""
import pickle

import numpy as np
import streamlit as st

DATA_DIR = "pose_estimation/data/"


@st.cache
def get_score(filepath):
    scores = pickle.load(open(filepath, "rb"))
    return np.mean(np.nanmean(scores, axis=1))


def pose_compare():
    st.title("Pose Comparison Demo")

    select_ex = st.selectbox("Select examples.", ["ex1", "ex2"])

    st.subheader("Video 1")
    st.video(DATA_DIR + f"{select_ex}/crop1.mp4")
    
    st.subheader("Video 2")
    st.video(DATA_DIR + f"{select_ex}/crop2.mp4")
    
    st.header("Pose Estimation of Target")
    st.write("A pose estimator model is used to detect the main body parts. "
             "The body parts form a connected graph.")
    
    st.subheader("Video 1: after pose estimation")
    st.video(DATA_DIR + f"{select_ex}/pose1.mp4")
    
    st.subheader("Video 2: after pose estimation")
    st.video(DATA_DIR + f"{select_ex}/pose2.mp4")
    
    st.header("Comparison Method")
    st.write("For each frame, we compare the (normalized) vectors formed by pairs of body parts "
             "using cosine similarity, which ranges from -1 (least similar) to 1 (most similar).")
    st.write("The body part pairs considered are ")
    st.write("""
        - RShoulder & RElbow
        - RElbow & RWrist
        - LShoulder & LElbow
        - LElbow & LWrist
        - RHip & RKnee
        - RKnee & RAnkle
        - LHip & LKnee
        - LKnee & LAnkle
        - Neck & RShoulder
        - Neck & LShoulder
        - Neck & RHip
        - Neck & LHip
    """)
    st.write("We then take the average of these similarity scores to get a mean score for each frame.")
    st.write("Finally, we take the average of these scores across frames.")
    
    score = get_score(DATA_DIR + f"{select_ex}/scores.pkl")
    st.write(f"**Mean similarity score = `{score:.4f}`**")
    

if __name__ == "__main__":
    pose_compare()
