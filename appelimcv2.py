import streamlit as st
import torch
import tempfile
from pathlib import Path
import moviepy.editor as mp
from io import BytesIO
from PIL import Image

# Load YOLOv5 model from GitHub and weights from Hugging Face
model = torch.hub.load(
    'https://github.com/CHANDU3094/TennisOpenCV',  # Full URL to the GitHub repository
    'custom', 
    path='https://huggingface.co/chandu3094/Streamlit/resolve/main/best.pt',  # Path to weights
    source='github'  # Indicate that the source is GitHub
)

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
            # Create a temporary file path to store the video content
            temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            # Write the BytesIO content to the temporary file
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(input_file.read())

            # Using MoviePy to read the video from the temporary file
            video = mp.VideoFileClip(temp_input_path)

            # Initialize a list to store processed frames
            processed_frames = []

            # Processing each frame
            progress_bar = st.progress(0)
            progress_label = st.empty()  # To display progress percentage
            for i, frame in enumerate(video.iter_frames(fps=video.fps, dtype="uint8")):
                # Convert frame to PIL image for YOLOv5 processing
                pil_image = Image.fromarray(frame)
                
                # Perform inference on the frame using YOLOv5
                results = model(pil_image)
                processed_frame = results.render()[0]

                # Append processed frame to list
                processed_frames.append(processed_frame)

                # Update progress bar
                progress_percentage = int(((i + 1) / video.reader.nframes) * 100)
                progress_bar.progress(progress_percentage)
                progress_label.text(f"Processing... {progress_percentage}% complete")

            # Create a new VideoClip from processed frames
            processed_clip = mp.ImageSequenceClip(processed_frames, fps=video.fps)

            # Write the processed video to output path
            processed_clip.write_videofile(temp_input_path, codec="libx264")

            # Set output path for download
            st.session_state.output_path = temp_input_path
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
