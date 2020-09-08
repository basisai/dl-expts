import streamlit as st

DATA_DIR = "detection/samples/"


def detect():
    st.title("Object Detection Demo")

    st.subheader("Methodologies")
    st.write("Objection detection algorithms:\n"
             "- YOLOv3, [YOLOv5](https://github.com/ultralytics/yolov5)\n"
             "- [EfficientDet](https://github.com/rwightman/efficientdet-pytorch)")
    st.write("Animal pose estimation:\n"
             "- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)\n"
             "- [DeepPoseKit](https://github.com/jgraving/DeepPoseKit)")
    st.write("Image annotation tools:\n"
             "- [CVAT](https://github.com/opencv/cvat)\n"
             "- [LabelImg](https://github.com/tzutalin/labelImg)")

    st.subheader("Example: after applying detection algorithm (YOLO)")
    st.video(DATA_DIR + "out2.mp4")
    

if __name__ == "__main__":
    detect()
