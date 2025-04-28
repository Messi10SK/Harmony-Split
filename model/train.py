import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader
from model.dataset_loader import MusicDataset
from model.custom_model import UNetSeparator
from utils.evaluation import compute_sdr





def train():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MusicDataset("data/processed")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = UNetSeparator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(10):
        model.train()
        for mixture, stems in dataloader:
            mixture, stems = mixture.to(device), stems.to(device)
            optimizer.zero_grad()
            pred_stems = model(mixture)
            loss = criterion(pred_stems, stems)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        sdr = compute_sdr(pred_stems, stems)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, SDR: {sdr:.4f}")

    # Save model
    os.makedirs("model/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "model/checkpoints/model.pth")

if __name__ == "__main__":
    train()