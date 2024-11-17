import streamlit as st
import cv2
import torch
import tempfile
from pathlib import Path
import requests

# Hugging Face URL for the model
model_url = "https://huggingface.co/chandu3094/Streamlit/resolve/main/best.pt"

@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

model = load_model()

# Streamlit page configuration
st.set_page_config(page_title="Tennis Game Tracking", layout="centered")

# Title of the application
st.title("Tennis Game Tracking")

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
            # Save the uploaded file temporarily
            temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_input_path, "wb") as f:
                f.write(input_file.read())

            # Open the input video
            cap = cv2.VideoCapture(temp_input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create a temporary output path
            st.session_state.output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(st.session_state.output_path, fourcc, fps, (width, height))

            # Processing each frame
            frame_num = 0
            progress_bar = st.progress(0)
            progress_label = st.empty()  # To display progress percentage

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform inference on the frame using YOLOv5
                results = model(frame)
                processed_frame = results.render()[0]

                # Write the processed frame to output
                out.write(processed_frame)

                # Update progress bar
                frame_num += 1
                progress_percentage = int((frame_num / total_frames) * 100)
                progress_bar.progress(progress_percentage)
                progress_label.text(f"Processing... {progress_percentage}% complete")

            cap.release()
            out.release()
            st.success("Video processing complete!")
            st.session_state.show_output_video = True  # Set to True to display output video

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
        st.video(st.session_state.output_path)  # Display the processed video
