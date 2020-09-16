import streamlit as st
from PIL import Image

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
    st.write("Image annotation tools:\n"
             "- [CVAT](https://github.com/opencv/cvat)\n"
             "- [LabelImg](https://github.com/tzutalin/labelImg)")

    st.subheader("Example: after applying detection algorithm (YOLO)")
    st.video(DATA_DIR + "out2.mp4")

    st.header("Pose estimation: prelims")
    st.write("We will first need to label the different parts of the elephant. Preliminary list of body parts are ")
    st.write("""
        - L_eye
        - R_eye
        - L_ear
        - R_ear
        - Throat
        - Tail
        - Withers
        - L_F_elbow
        - R_F_elbow
        - L_B_elbow
        - R_B_elbow
        - L_F_knee
        - R_F_knee
        - L_B_knee
        - R_B_knee
        - L_F_paw
        - R_F_paw
        - L_B_paw
        - R_B_paw
        - Mid_trunk
        - End_trunk
    """)
    st.write("`Do they look ok?`")

    st.subheader("Example of labelling an image frame")
    st.image(DATA_DIR + "frame146_anno.jpeg")

    st.subheader("Another example: horse")
    st.image(DATA_DIR + "ho1_anno.jpeg")


if __name__ == "__main__":
    detect()
