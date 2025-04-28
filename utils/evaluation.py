import numpy as np
import museval

def compute_sdr(pred_stems, true_stems):
    """Compute Signal-to-Distortion Ratio (SDR) using museval."""
    pred_stems = pred_stems.detach().cpu().numpy()
    true_stems = true_stems.detach().cpu().numpy()
    
    # Dummy evaluation (replace with real museval call for actual dataset)
    sdr = np.mean([10 * np.log10(np.var(true) / np.var(true - pred)) 
                   for pred, true in zip(pred_stems, true_stems)])
    return sdr