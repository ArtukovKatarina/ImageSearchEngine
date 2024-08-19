from pathlib import Path
import torch

ROOT = Path(__file__).absolute().parent.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "flickr30k_images"
PROCESSED_PATH = ROOT / "data" / "processed"

DEBUG = False
BATCH_SIZE = 32
NUM_WORKERS = 4
HEAD_LR = 1e-3
IMAGE_ENCODER_LR = 1e-4
TEXT_ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 1
FACTOR = 0.8
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECTION_DIM = 256 
DROPOUT = 0.1

MAX_LENGTH = 200
IMAGE_SIZE = 224

MODEL_NAME = 'resnet50'
IMAGE_EMBEDDING = 2048
TEXT_ENCODER_MODEL = "distilbert-base-uncased"
TEXT_EMBEDDING = 768
TEXT_TOKENIZER = "distilbert-base-uncased"

PRETRAINED = True
TRAINABLE = True 
TEMPERATURE = 1.0