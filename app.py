import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp 
from typing import Optional 
from mediapipe.framework.formats import landmark_pb2 as mp_landmark 


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

def calc_joint_angle(point1, point2, point3):
    """
    Calculate angle at point2 (the joint)
    point1, point2, point3: tuples of (x, y) coordinates - point2 is equal to the vertex of the triangle where the angle is measured. 
    """
    # Create vectors from joint to adjacent points
    vec1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vec2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    print(vec1.shape)
    print(vec2.shape)
    
    # Calculate angle
    dot_prod = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    angle = np.arccos(np.clip(dot_prod / (mag1 * mag2), -1.0, 1.0))
    
    return np.degrees(angle)


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
        key = "video_file",
    )
if uploaded is None and "video_path" not in st.session_state: 
    st.info("Upload a video to begin.")
    st.stop()

if uploaded is not None and uploaded.name != st.session_state.get("uploaded_name"):
    # Save a temp file so OpenCv can access it
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix = suffix) as tmp:
        tmp.write(uploaded.read())
        st.session_state.video_path = tmp.name
        st.session_state.uploaded_name = uploaded.name
        
        video_path = Path(st.session_state.video_path)
    
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
        with col2: 
            if st.button("Next Frame", width ="stretch"):
                st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                st.session_state.playing = False 

        selected_frame = int(st.session_state.current_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()
        cap.release()
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

            LSHO = np.array([landmark_df["LEFT_SHOULDER_x"], landmark_df["LEFT_SHOULDER_y"]]).flatten()
            LHIP = np.array([landmark_df["LEFT_HIP_x"], landmark_df["LEFT_HIP_y"]]).flatten()
            LKNEE = np.array([landmark_df["LEFT_KNEE_x"], landmark_df["LEFT_KNEE_y"]]).flatten()
            LANKLE = np.array([landmark_df["LEFT_ANKLE_x"], landmark_df["LEFT_ANKLE_y"]]).flatten()
            LTOE = np.array([landmark_df["LEFT_FOOT_INDEX_x"], landmark_df["LEFT_FOOT_INDEX_y"]]).flatten()

            RSHO = np.array([landmark_df["RIGHT_SHOULDER_x"], landmark_df["RIGHT_SHOULDER_y"]]).flatten()
            RHIP = np.array([landmark_df["RIGHT_HIP_x"], landmark_df["RIGHT_HIP_y"]]).flatten()
            RKNEE = np.array([landmark_df["RIGHT_KNEE_x"], landmark_df["RIGHT_KNEE_y"]]).flatten()
            RANKLE = np.array([landmark_df["RIGHT_ANKLE_x"], landmark_df["RIGHT_ANKLE_y"]]).flatten()
            RTOE = np.array([landmark_df["RIGHT_FOOT_INDEX_x"], landmark_df["RIGHT_FOOT_INDEX_y"]]).flatten()

            L_hip_angle= calc_joint_angle(LSHO, LHIP, LKNEE)
            L_knee_angle = calc_joint_angle(LHIP,LKNEE,LANKLE)
            L_ankle_angle = calc_joint_angle(LKNEE, LANKLE, LTOE)

            R_hip_angle= calc_joint_angle(RSHO, RHIP, RKNEE)
            R_knee_angle = calc_joint_angle(RHIP,RKNEE,RANKLE)
            R_ankle_angle = calc_joint_angle(RKNEE, RANKLE, RTOE)


            #columns_to_display = ["LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y", "RIGHT_HIP_x", "RIGHT_HIP_y", "LEFT_HIP_x", "LEFT_HIP_y", "LEFT_KNEE_x", "LEFT_KNEE_y", "RIGHT_KNEE_x", "RIGHT_KNEE_y", "LEFT_ANKLE_x", "LEFT_ANKLE_y", "RIGHT_ANKLE_x","RIGHT_ANKLE_y"]

            #st.dataframe(landmark_df[columns_to_display].head(10), use_container_width=True)
            colsJA = ["Left Hip", "Left Knee", "Left Ankle"]
            JA_df = pd.DataFrame({
                'Joint': ['Hip', 'Knee', 'Ankle'],
                'Left': [L_hip_angle, L_knee_angle, L_ankle_angle],
                'Right': [R_hip_angle, R_knee_angle, R_ankle_angle]
            })

            st.write(f"Frame: {selected_frame} | Time: {selected_frame / fps:.2f}s")
            st.dataframe(JA_df[['Joint', 'Left', 'Right']].head(10), use_container_width=True)






            

                       





               
