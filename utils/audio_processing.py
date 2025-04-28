import librosa
import soundfile as sf
import ffmpeg
import numpy as np
import os

def preprocess_audio(file_path):
    """Convert to 44100Hz WAV and return audio data."""
    output_path = f"data/processed/{os.path.basename(file_path).split('.')[0]}.wav"
    os.makedirs("data/processed", exist_ok=True)
    
    # Convert to WAV using ffmpeg
    stream = ffmpeg.input(file_path)
    stream = ffmpeg.output(stream, output_path, ar=44100, ac=1, format="wav")
    ffmpeg.run(stream, quiet=True)
    
    # Load audio
    audio, sr = librosa.load(output_path, sr=44100, mono=True)
    return audio, sr

def save_stems(stems, output_dir, sr):
    """Save separated stems as WAV files."""
    os.makedirs(output_dir, exist_ok=True)
    stem_names = ["vocals", "drums", "bass", "other"]
    for i, stem in enumerate(stems):
        stem_path = f"{output_dir}/{stem_names[i]}.wav"
        sf.write(stem_path, stem.cpu().numpy(), sr)