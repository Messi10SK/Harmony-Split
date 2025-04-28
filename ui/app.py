import streamlit as st
import os
from utils.audio_processing import preprocess_audio, save_stems
from model.custom_model import UNetSeparator
import torch

st.set_page_config(page_title="Harmony-Split", page_icon="🎵")
st.title("Harmony-Split: Music Source Separation")
st.write("Upload an audio file to separate it into vocals, drums, bass, and other stems.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file:
    # Save uploaded file
    input_path = f"data/raw/{uploaded_file.name}"
    os.makedirs("data/raw", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process audio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load("model/checkpoints/model.pth"))
    model.eval()

    audio, sr = preprocess_audio(input_path)
    audio = torch.tensor(audio, dtype=torch.float32).to(device)

    with torch.no_grad():
        stems = model(audio.unsqueeze(0))
    
    # Save and display stems
    output_dir = f"data/stems/{uploaded_file.name.split('.')[0]}"
    save_stems(stems.squeeze(0), output_dir, sr)
    
    st.success("Separation complete! Download stems below:")
    for stem_name in ["vocals", "drums", "bass", "other"]:
        stem_path = f"{output_dir}/{stem_name}.wav"
        st.audio(stem_path)
        with open(stem_path, "rb") as f:
            st.download_button(f"Download {stem_name}.wav", f, file_name=f"{stem_name}.wav")