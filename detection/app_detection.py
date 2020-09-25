import streamlit as st

DATA_DIR = "detection/samples/"


def detect():
    st.title("Object Detection Demo")

    st.subheader("Methodologies")
    st.write("Objection detection algorithms:\n"
             "- YOLOv4, [YOLOv5](https://github.com/ultralytics/yolov5)\n"
             "- [EfficientDet](https://github.com/rwightman/efficientdet-pytorch)")
    st.write("Animal pose estimation:\n"
             "- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)\n"
             "- [DeepPoseKit](https://github.com/jgraving/DeepPoseKit)")

    st.subheader("Data")
    st.write("To train the models, we will need labelled data. "
             "We will collect video data, extract the frames and hand-label them:\n"
             "- Bounding boxes (for detecting elephants)\n"
             "- Keypoints (for detecting elephant body parts)")
    st.write("To help in labelling, we will use image annotation tools such as\n"
             "- [CVAT](https://github.com/opencv/cvat)\n"
             "- [LabelImg](https://github.com/tzutalin/labelImg)")

    st.header("Elephant detection")
    st.write("From `C1`, we extract some images, label the elephants in them, and train a YOLO model.")
    st.subheader("After applying YOLO detection algorithm")
    st.write("**Example 1**")
    st.video(DATA_DIR + "sample1_infer.mp4")
    st.write("**Example 2**")
    st.video(DATA_DIR + "sample2_infer.mp4")

    st.header("Pose estimation: prelims")
    st.write("We will first need to label the keypoints of the elephant. Preliminary list of keypoints:")
    st.write("""
        - L_eye, R_eye
        - L_ear, R_ear
        - Nose, Mid_trunk, End_trunk
        - Throat
        - Tail
        - Withers
        - L_F_elbow, R_F_elbow, L_B_elbow, R_B_elbow
        - L_F_knee, R_F_knee, L_B_knee, R_B_knee
        - L_F_paw, R_F_paw, L_B_paw, R_B_paw
    """)

    st.subheader("Example of labelling an image frame")
    st.image(DATA_DIR + "frame146_anno.jpeg")


if __name__ == "__main__":
    detect()
