import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --- Optional: choose which MediaPipe solution you want ---
# Pose is shown here; you can switch to Holistic/Face Mesh/Hands similarly
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# --- Explicitly import the landmark proto to avoid lazy-import attribute issues
from typing import Optional
from mediapipe.framework.formats import landmark_pb2 as mp_landmark

st.set_page_config(page_title="MediaPipe Video Processor", layout="wide")
st.title("Process an uploaded video with MediaPipe Pose")

with st.sidebar:
    st.header("Settings")
    model_complexity = st.select_slider("Model complexity", options=[0, 1, 2], value=1,
                                        help="0 = light, 2 = most accurate")
    detection_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
    tracking_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)
    enable_seg = st.checkbox("Enable segmentation mask", value=False)
    draw_landmarks = st.checkbox("Draw landmarks on output video", value=True)

uploaded = st.file_uploader(
    "Upload a video (mp4/mov/avi/mkv)",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)

@st.cache_data(show_spinner=False)
def _pose_columns():
    cols = [f"frame"]
    for lm in mp_pose.PoseLandmark:
        cols += [f"{lm.name}_x", f"{lm.name}_y", f"{lm.name}_z", f"{lm.name}_v"]
    return cols


def _extract_row(frame_idx: int, results: Optional[mp_landmark.NormalizedLandmarkList]):
    row = [frame_idx]
    if results and results.landmark:
        for lm in results.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        # If no landmarks this frame, fill with NaNs
        row.extend([np.nan] * (len(_pose_columns()) - 1))
    return row


if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

# --- Save the uploaded bytes to a temp file so OpenCV can read it ---
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp_in:
    tmp_in.write(uploaded.read())
    in_path = Path(tmp_in.name)

cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened():
    st.error("Could not open video. Try a different file or codec.")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Prepare output video writer ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = Path(tempfile.mkstemp(suffix=".mp4")[1])
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

# --- Landmark table ---
cols = _pose_columns()
rows = []

progress = st.progress(0.0, text="Processing…")
status = st.empty()

# --- Process frames with MediaPipe Pose ---
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=model_complexity,
    enable_segmentation=enable_seg,
    min_detection_confidence=detection_conf,
    min_tracking_confidence=tracking_conf,
) as pose:
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Collect landmarks for CSV
        if results.pose_landmarks:
            rows.append(_extract_row(i, results.pose_landmarks))
        else:
            rows.append(_extract_row(i, None))

        # Draw output
        if draw_landmarks:
            annotated = frame_bgr.copy()
            if enable_seg and results.segmentation_mask is not None:
                # Simple overlay to show segmentation mask
                mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                annotated = cv2.addWeighted(annotated, 0.7, mask_col, 0.3, 0)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
        else:
            annotated = frame_bgr

        writer.write(annotated)

        i += 1
        if frame_count > 0:
            progress.progress(min(i / frame_count, 1.0), text=f"Processing frame {i}/{frame_count}")
        else:
            progress.progress((i % 100) / 100.0)

cap.release()
writer.release()

progress.empty()
status.info(f"Done. Processed {i} frames at ~{fps:.1f} FPS, output size: {width}×{height}.")

# --- Show output video and downloads ---
st.subheader("Results")
with open(out_path, "rb") as f:
    st.video(f.read())

# Build DataFrame and provide download
landmarks_df = pd.DataFrame(rows, columns=cols)
st.caption("Pose landmarks per frame (x, y, z, visibility for each joint).")
st.dataframe(landmarks_df.head(10), use_container_width=True)

csv_bytes = landmarks_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download landmarks CSV",
    data=csv_bytes,
    file_name=f"{Path(uploaded.name).stem}_pose_landmarks.csv",
    mime="text/csv",
)

st.download_button(
    "Download annotated video (.mp4)",
    data=open(out_path, "rb").read(),
    file_name=f"{Path(uploaded.name).stem}_annotated.mp4",
    mime="video/mp4",
)

st.success("All done! You can tweak settings in the sidebar and re-upload another video.")

st.markdown(
    """
**Tips**
- Record your video perpendicular to the sagital plane
- Keep your video short (e.g. Less then 10 seconds) 
    """
)
