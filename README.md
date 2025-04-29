# Harmony-Split

A music source separation tool to split audio into vocals, drums, bass, and other stems using a custom UNet-based model.

## Setup
1. Clone the repo: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Install ffmpeg: `sudo apt-get install ffmpeg` (or equivalent)
4. Run the Streamlit app: `streamlit run ui/app.py`

## Directory Structure
- `data/`: Raw, processed audio, and stems
- `model/`: Training scripts, model architecture, and checkpoints
- `ui/`: Streamlit web app
- `utils/`: Audio processing, evaluation, and configs
- `deployment/`: Docker setup

## Usage
- Upload an audio file via the web app.
- The model processes it and outputs separated stems.
- Evaluate separation quality using SDR/SIR metrics.

## Training
Run `python model/train.py` to train the model.

## ScreenShots
![Alt Text](images/before.png)

![Alt Text](images/After.png)


## License
MIT