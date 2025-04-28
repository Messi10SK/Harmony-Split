import os
from utils.audio_processing import preprocess_audio, save_stems
from model.custom_model import UNetSeparator
import torch

def main(audio_path, output_dir):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load("model/checkpoints/model.pth"))
    model.eval()

    # Process audio
    audio, sr = preprocess_audio(audio_path)
    audio = torch.tensor(audio, dtype=torch.float32).to(device)

    # Separate sources
    with torch.no_grad():
        stems = model(audio.unsqueeze(0))  # [vocals, drums, bass, other]
    
    # Save stems
    save_stems(stems.squeeze(0), output_dir, sr)

if __name__ == "__main__":
    input_audio = "data/raw/sample.mp3"
    output_dir = "data/stems/sample"
    os.makedirs(output_dir, exist_ok=True)
    main(input_audio, output_dir)