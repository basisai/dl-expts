"""
Streamlit app
"""
import streamlit as st

DATA_DIR = "detection/samples/"


def detect():
    """App."""
    st.title("Object Detection and Tracking Demo")

    st.header("Elephant detection")
    st.write(
        """
        From the video data, we extract some images, label the bounding boxes of the elephants,
        and train a detection model.
        """
    )
    st.write("**Example 1**")
    st.video(DATA_DIR + "sample1_infer.mp4")
    st.write("**Example 2**")
    st.video(DATA_DIR + "sample2_infer.mp4")

    st.header("Elephant Tracking")
    st.write(
        """
        The detections from the multi-camera setup are fused together before applying a tracking
        algorithm to track the elephants.
        """
    )
    st.video(DATA_DIR + "overlay.mp4")

    st.header("Pose estimation: prelims")
    st.write(
        """
        Animal pose estimation:
        - [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
        - [DeepPoseKit](https://github.com/jgraving/DeepPoseKit)

        We will first need to label the keypoints of the elephant. Preliminary list of keypoints:
        - L_eye, R_eye
        - L_ear, R_ear
        - Nose, Mid_trunk, End_trunk
        - Throat
        - Tail
        - Withers
        - L_F_elbow, R_F_elbow, L_B_elbow, R_B_elbow
        - L_F_knee, R_F_knee, L_B_knee, R_B_knee
        - L_F_paw, R_F_paw, L_B_paw, R_B_paw
        """
    )

    st.subheader("Example of labelling an image frame")
    st.image(DATA_DIR + "frame146_anno.jpeg")
    
    st.header("Annex")
    st.subheader("Data")
    st.write(
        """
        Image annotation tools
        - [CVAT](https://github.com/opencv/cvat)
        - [LabelImg](https://github.com/tzutalin/labelImg)

        Objection detection algorithms:
        - YOLOv4, [YOLOv5](https://github.com/ultralytics/yolov5)
        - [EfficientDet](https://github.com/rwightman/efficientdet-pytorch)
        """
    )


if __name__ == "__main__":
    detect()
