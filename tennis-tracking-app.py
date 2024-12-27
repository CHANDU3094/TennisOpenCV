import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Define local paths for the model and repository (update for your environment)
model_path = 'best.pt'  # Replace with your actual .pt file path

# Attempt to load the custom YOLOv5 model
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

# Main App UI
st.title('🎾 Tennis Tracking App')

# "HOW TO USE?" Button
if st.button("HOW TO USE?"):
    st.markdown(
        """
        **Welcome to the Tennis Tracking App! 🎾**

        **Step 1: Upload Your Video**
        🔍 Choose a tennis video (MP4, AVI, or MOV format) from your device. The app supports various formats to get started!

        **Step 2: Let the Magic Happen**
        ⏭️ Once uploaded, the app will analyze the video and track the players in real-time. Sit back, relax, and watch as your video gets processed!

        **Step 3: Download Your Processed Video**
        ⬇️ After processing, you can download the video with all the detected player tracking included. Ready for sharing or further analysis!

        **Happy Tracking!**
        """
    )

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("⏳ Processing video... Please wait.")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model (mixed precision if CUDA is available)
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write frame to output video file
        out.write(frame)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate in display
        time.sleep(1 / fps)

    # Release video resources
    cap.release()
    out.release()

    st.success("Video Processing Completed Successfully✅")

    # Provide download button for the processed video
    st.write("📥 Download the processed video:")
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="📥 Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
