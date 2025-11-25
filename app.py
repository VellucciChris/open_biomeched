import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp 
from typing import Optional 
from mediapipe.framework.formats import landmark_pb2 as mp_landmark 


# Cache the video file saving to avoid re-processing on every interaction
@st.cache_data
def save_uploaded_video(uploaded_file):
    """Save uploaded video to temp file and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        return Path(tmp.name)


# Cache video properties to avoid reopening the video file repeatedly
@st.cache_data
def get_video_properties(_video_path):
    """Extract video properties without keeping the capture object open"""
    cap = cv2.VideoCapture(str(_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': total_frames / fps if fps > 0 else 0
    }


# DON'T cache frames - read on demand to save memory
def read_frame(video_path, frame_number, max_width=1200):
    """Read a specific frame from video with optional downsampling"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Resize if too large to reduce memory usage
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (max_width, new_h))
        return frame
    return None


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

    uploaded = st.file_uploader(
        "Upload a video (mp4/mov/avi/mkv)",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )

if uploaded is None: 
    st.info("Upload a video to begin.")
    st.stop()

if uploaded: 
    # Use cached function to save video (only processes once per unique file)
    video_path = save_uploaded_video(uploaded)
    
    with st.container(): 
        st.header("Step 2 - Watch the Video")
        st.video(str(video_path))

    
    with st.container(): 
        st.header("Step 3 - Select Frame and Perform Pose Estimation")

        # Use cached function to get video properties (only processes once)
        video_props = get_video_properties(video_path)
        total_frames = video_props['total_frames']
        fps = video_props['fps']
        duration = video_props['duration']

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
            if st.button("Previous Frame", use_container_width=True):
                st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                st.session_state.playing = False
                st.rerun()
        with col2: 
            if st.button("Next Frame", use_container_width=True):
                st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                st.session_state.playing = False 
                st.rerun()

        selected_frame = int(st.session_state.current_frame)

        # Use cached function to read frame (caches each frame independently)
        frame = read_frame(video_path, selected_frame)
        
        if frame is not None: 
            st.image(frame, channels="BGR", caption = f"Frame {selected_frame}")
            st.session_state['paused_frame'] = selected_frame
        else:
            st.error("Couldn't read the frame.")

        run_pose = st.button("Perform Pose Estimation", type="primary", use_container_width=True)
        
        if run_pose and frame is not None: 
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

            st.image(output, channels="BGR", caption = f"Frame {selected_frame} Pose")
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

            L_hip_angle = calc_joint_angle(LSHO, LHIP, LKNEE)
            L_knee_angle = calc_joint_angle(LHIP, LKNEE, LANKLE)
            L_ankle_angle = calc_joint_angle(LKNEE, LANKLE, LTOE)

            R_hip_angle = calc_joint_angle(RSHO, RHIP, RKNEE)
            R_knee_angle = calc_joint_angle(RHIP, RKNEE, RANKLE)
            R_ankle_angle = calc_joint_angle(RKNEE, RANKLE, RTOE)

            JA_df = pd.DataFrame({
                'Joint': ['Hip', 'Knee', 'Ankle'],
                'Left': [L_hip_angle, L_knee_angle, L_ankle_angle],
                'Right': [R_hip_angle, R_knee_angle, R_ankle_angle]
            })

            st.write(f"Frame: {selected_frame} | Time: {selected_frame / fps:.2f}s")
            st.dataframe(JA_df[['Joint', 'Left', 'Right']].head(10), use_container_width=True)