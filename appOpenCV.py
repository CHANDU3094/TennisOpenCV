import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Load YOLOv5 model
model_path = 'best.pt'
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🎾 Tennis Player Tracking App")

# File uploader
uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Output video setup
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = np.squeeze(results.render())
        out.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

        progress_bar.progress(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / total_frames)

        time.sleep(1 / fps)

    cap.release()
    out.release()

    with open(output_video_path, 'rb') as f:
        st.download_button("Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")

    os.remove(temp_video_path)
    os.remove(output_video_path)
