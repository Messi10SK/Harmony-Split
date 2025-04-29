import streamlit as st
import os
import librosa
import torch
import gc
import numpy as np
import soundfile as sf
import sys
import os

# Add the parent directory to the system path so 'model' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.custom_model import UNetSeparator  # Import your model from the 'model' directory
from utils.audio_processing import save_stems  # Assuming you have this utility function

# Set Streamlit page configuration
st.set_page_config(page_title="Harmony-Split", page_icon="🎵")
st.title("Harmony-Split: Music Source Separation and Karaoke Creation")
st.write("Upload an audio file to separate it into vocals, drums, bass, and other stems, and generate a karaoke version.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

def preprocess_audio(input_path, target_sr=16000):
    """Preprocess audio by loading it."""
    # Load the full audio file without limiting its duration
    audio, sr = librosa.load(input_path, sr=target_sr)
    return audio, sr

def generate_karaoke(instrumentals, vocals):
    """Generate karaoke by subtracting vocals from the instrumentals."""
    karaoke = instrumentals - vocals
    # Ensure the karaoke track stays within [-1, 1]
    karaoke = np.clip(karaoke, -1, 1)
    return karaoke

if uploaded_file:
    # Save uploaded file to a specific location
    input_path = f"data/raw/{uploaded_file.name}"
    os.makedirs("data/raw", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Set up the model for inference
    device = torch.device("cpu")  # Use CPU
    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load("model/checkpoints/model.pth"))
    model.eval()

    # Preprocess the audio
    audio, sr = preprocess_audio(input_path)
    audio = torch.tensor(audio, dtype=torch.float32).to(device)

    # Perform source separation
    with torch.no_grad():
        stems = model(audio.unsqueeze(0))

    # Save and display the stems
    output_dir = f"data/stems/{uploaded_file.name.split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)
    save_stems(stems.squeeze(0), output_dir, sr)
    
    # Extract vocals and instrumentals
    vocals = stems[0].cpu().numpy()  # Assuming vocals are the first stem
    instrumentals = np.sum(stems[1:].cpu().numpy(), axis=0)  # Sum the remaining stems for instrumentals

    # Display stem info
    st.write(f"Vocals max value: {np.max(vocals)}, min value: {np.min(vocals)}")
    st.write(f"Instrumentals max value: {np.max(instrumentals)}, min value: {np.min(instrumentals)}")

    # Check if the stems are silent
    if np.max(vocals) == 0 or np.max(instrumentals) == 0:
        st.warning("The extracted vocals or instrumentals are silent. Please check your model output.")

    # Generate karaoke by subtracting vocals from instrumentals
    karaoke = generate_karaoke(instrumentals, vocals)
    
    # Ensure the karaoke track is audible and normalized
    st.write(f"Karaoke max value before clipping: {np.max(karaoke)}")
    karaoke = np.clip(karaoke, -1, 1)
    st.write(f"Karaoke max value after clipping: {np.max(karaoke)}")

    # Save the generated karaoke track
    karaoke_path = f"{output_dir}/karaoke.wav"
    sf.write(karaoke_path, karaoke, sr)  # Using soundfile to write the .wav file
    
    st.success("Separation complete! Download stems and karaoke below:")

    # Provide download buttons for each stem and the karaoke track
    for stem_name in ["vocals", "drums", "bass", "other"]:
        stem_path = f"{output_dir}/{stem_name}.wav"
        st.audio(stem_path)
        with open(stem_path, "rb") as f:
            st.download_button(f"Download {stem_name}.wav", f, file_name=f"{stem_name}.wav")
    
    # Provide download button for the karaoke track
    with open(karaoke_path, "rb") as f:
        st.download_button("Download Karaoke Track", f, file_name="karaoke.wav")

    # Free up memory after processing
    del audio, vocals, instrumentals, karaoke
    gc.collect()
