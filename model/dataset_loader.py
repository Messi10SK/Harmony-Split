import torch
from torch.utils.data import Dataset
import os
import librosa
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, data_dir, slice_duration=5):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.slice_duration = slice_duration  # seconds
        self.sample_rate = 44100

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        max_samples = self.slice_duration * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]  # Keep only first 5 seconds

        # Simulate mixture and stems (replace with real stems later)
        mixture = audio  # mono mixture
        stems = np.stack([audio, audio, audio, audio], axis=0)  # dummy stems

        return torch.tensor(mixture, dtype=torch.float32), torch.tensor(stems, dtype=torch.float32).squeeze(0)

