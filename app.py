import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp 
from typing import Optional 
from mediapipe.framework.formats import landmark_pb2 as mp_landmark 

import platform, subprocess, sys
import streamlit as st
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())

# What libs are present?
try:
    out = subprocess.check_output(["ldconfig", "-p"]).decode()
    st.code("\n".join([l for l in out.splitlines() if "libGL.so.1" in l or "libGL" in l][:20]))
except Exception as e:
    st.write("ldconfig check failed:", e)

# Which OpenCV did we import?
import cv2, pathlib
st.write("cv2 file:", pathlib.Path(cv2.__file__).as_posix())
st.write("OpenCV version:", cv2.__version__)



def update_slider():
    st.session_state.current_frame = st.session_state.frame
    st.session_state.playing = False

def pose_columns(): 
    cols = [f"frame"]
    for lm in mp_pose.PoseLandmark: 
        cols += [f"{lm.name}_x", f"{lm.name}_y", f"{lm.name}_z", f"{lm.name}_v"]
    return cols

def pose_row(frame_idx: int, results :Optional[mp_landmark.NormalizedLandmarkList]):
    row = [frame_idx]
    if results and results.landmark: 
        for lm in results.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else: 
        row.extend([np.nan] * (len(pose_columns())-1))
    return row
                



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

st.set_page_config(page_title="MediaPipe Video Processor", layout="wide")
st.title("Lab 5 - Using AI Motion Analysis To Assess Squat Technique")

with st.sidebar:
    st.header("Settings")
    model_complexity = st.select_slider("Model complexity", options=[0, 1, 2], value=1,
                                        help="0 = light, 2 = most accurate")
    detection_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    tracking_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    enable_seg = st.checkbox("Enable segmentation mask", value=False)
    draw_landmarks = st.checkbox("Draw landmarks on output video", value=True)


with st.container(): 
    st.header("Step 1 - Upload a Video")
    if "video_path" not in st.session_state:
        st.session_state.video_path = None

    uploaded = st.file_uploader(
        "Upload a video (mp4/mov/avi/mkv)",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )
if uploaded is None: 
    st.info("Upload a video to begin.")
    st.stop()

if uploaded: 
    # Save a temp file so OpenCv can access it
    with tempfile.NamedTemporaryFile(delete=False, suffix = Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = Path(tmp.name)
    
    with st.container(): 
        st.header("Step 2 - Watch the Video")
        st.video(video_path)

    
    with st.container(): 
        st.header("Step 3 - Select Frame and Perform Pose Estimation")

        # Load Video Info 
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames/fps


        if "current_frame" not in st.session_state: 
            st.session_state.current_frame = 0
        if "playing" not in st.session_state: 
            st.session_state.playing = False
        
        # Show Selected Frame 
        # Slider to "select" a frame 
        st.slider("Select a frame", 0, total_frames-1, 
                   value = int(st.session_state.current_frame), 
                   key="frame",
                   on_change = update_slider,
                   )
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Previous Frame", width = "stretch"):
                st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                st.session_state.playing = False
                st.rerun()
        with col2: 
            if st.button("Next Frame", width ="stretch"):
                st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                st.session_state.playing = False 
                st.rerun()

        selected_frame = int(st.session_state.current_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()

        if ret: 
            st.image(frame, channels="BGR", caption = f"Frame {selected_frame}")
            st.session_state['paused_frame'] = selected_frame
        else:
            st.error("Couldn't read the frame.") 
        cap.release()

        run_pose = st.button("Perform Pose Estimation", type ="primary", use_container_width=True)
        
        if run_pose: 
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_pose.Pose(static_image_mode=True,
                              model_complexity = model_complexity, 
                              enable_segmentation = enable_seg, 
                              min_detection_confidence = detection_conf, 
                              min_tracking_confidence = tracking_conf,
            ) as pose:
                results = pose.process(image_rgb)
            
            output = frame.copy()

            if results.pose_landmarks: 
                mp_drawing.draw_landmarks(
                    output, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS, 
                    landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
                )
                cols = pose_columns()
                row = pose_row(0, results.pose_landmarks)

            
            landmark_df = pd.DataFrame([row], columns=cols)

            st.image(output,channels="BGR", caption = f"Frame {selected_frame} Pose")
            st.caption("Pose landmarks per frame (x, y, z, visibility for each joint).")
            columns_to_display = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y", "RIGHT_HIP_x", "RIGHT_HIP_y", "LEFT_HIP_x", "LEFT_HIP_y", "LEFT_KNEE_x", "LEFT_KNEE_y", "RIGHT_KNEE_x", "RIGHT_KNEE_y", "LEFT_ANKLE_x", "LEFT_ANKLE_y", "RIGHT_ANKLE_x","RIGHT_ANKLE_y"]

            st.dataframe(landmark_df[columns_to_display].head(10), use_container_width=True)

            
