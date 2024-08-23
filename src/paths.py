from pathlib import Path
import torch

ROOT = Path(__file__).absolute().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "flickr30k_images"
PROCESSED_PATH = ROOT / "data" / "processed"
MODELS_PATH = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")