# app.py
import streamlit as st
import torch
import tempfile
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2
import numpy as np
import requests
from torchvision.transforms import functional as F

# Streamlit page configuration
st.set_page_config(page_title="Tennis Game Tracking", layout="centered")

# Title of the application
st.title("Tennis Game Tracking")

# Hugging Face model URL
model_url = "https://huggingface.co/chandu3094/Streamlit/resolve/main/best.torchscript"

@st.cache_resource
def load_model():
    # Load the model from Hugging Face
    model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
    response = requests.get(model_url)
    response.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(response.content)
    model = torch.jit.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

# Load YOLOv5 model
model = load_model()
model.eval()  # Ensure the model is in evaluation mode

# Initialize flags and file paths
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'input_file_name' not in st.session_state:
    st.session_state.input_file_name = None
if 'show_input_video' not in st.session_state:
    st.session_state.show_input_video = False
if 'show_output_video' not in st.session_state:
    st.session_state.show_output_video = False

# Layout setup: video display area on the left, buttons on the right
col1, col2 = st.columns([10, 7])

# File uploader and buttons on the right side
with col2:
    # File uploader for selecting input file
    input_file = st.file_uploader("Select Input File", type=["mp4", "mov", "avi"])

    # Set input file name if a file is uploaded
    if input_file:
        st.session_state.input_file_name = Path(input_file.name).stem  # Extract the file name without extension

    # Preview button
    if st.button("Preview Video"):
        if input_file:
            st.session_state.show_input_video = True
        else:
            st.warning("Please select a video file to preview.")

    # Process Video button
    if st.button("Process Video"):
        if input_file:
            # Create a temporary file path to store the video content
            temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_input_path, "wb") as f:
                f.write(input_file.read())

            cap = cv2.VideoCapture(temp_input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create output video writer
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            progress_label = st.empty()

            # Transformation: Resize to 640x640 and convert to tensor
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to PIL Image and transform
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_tensor = transform(pil_image).unsqueeze(0)

                # Perform inference
                with torch.no_grad():
                    results = model(image_tensor)

                # Render results on the frame
                if hasattr(results, 'render'):
                    processed_frame = np.squeeze(results.render())  # Use render if available
                else:
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Fallback to original frame
                
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
                out.write(processed_frame)

                frame_count += 1
                progress_percentage = int((frame_count / total_frames) * 100)
                progress_bar.progress(progress_percentage)
                progress_label.text(f"Processing... {progress_percentage}% complete")

            cap.release()
            out.release()

            # Set output path for download
            st.session_state.output_path = temp_output_path
            st.success("Video processing complete!")
            st.session_state.show_output_video = True

        else:
            st.warning("Please select a video file to process.")

    # Show Output button
    if st.button("Show Output"):
        if st.session_state.output_path:
            st.session_state.show_output_video = True
        else:
            st.warning("Please process the video before showing the output.")

    # Download Output button
    st.write("Download Output:")
    if st.session_state.output_path and Path(st.session_state.output_path).exists():
        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=f"{st.session_state.input_file_name}_output.mp4",
                mime="video/mp4"
            )
    else:
        st.warning("No processed video available. Please upload and process a video first.")

# Video display area in the larger left column
with col1:
    # Show input video if it has been previewed
    if st.session_state.show_input_video:
        st.subheader("Input Video Preview:")
        st.video(input_file)

    # Show processed output video below the input video preview
    if st.session_state.show_output_video and st.session_state.output_path:
        st.subheader("Processed Output Video:")
        st.video(st.session_state.output_path)
