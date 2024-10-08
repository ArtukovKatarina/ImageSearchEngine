import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import logging
from pinecone import Pinecone, ServerlessSpec
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer
from config import settings
from src import paths
from src.model.clip import Clip
from src.train.train import build_loaders, make_train_valid_dfs
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def init_pinecone(pinecone):
    INDEX_NAME = "clip-image-text-search"

    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    pinecone.create_index(name=INDEX_NAME, 
        dimension=settings.projection_dim, 
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))

    index = pinecone.Index(INDEX_NAME)
    logger.info("Created index with name: %s ", INDEX_NAME)
    

def get_image_embeddings(valid_df, model_path):
    logger.info("Start getting embeddings from CLIP.")
    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = Clip(
        settings.temperature,
        settings.image_embedding,
        settings.text_embedding,
        settings.model_name,
        settings.text_encoder_model,
        settings.pretrained,
        settings.trainable,
        settings.projection_dim,
        settings.dropout
        ).to(paths.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=paths.DEVICE))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(paths.DEVICE))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    
    logger.info("Finish getting embeddings from CLIP.")
    return model, torch.cat(valid_image_embeddings)

def store_image_embeddings(pinecone, valid_df, image_embeddings, image_filenames):
    logger.info("Start storing embeddings to Pinecone.")
    logger.info("Image embeddings number: %f", len(image_embeddings))
    index = pinecone.Index("clip-image-text-search")
    BATCH_SIZE = 250
    # Prepare data for Pinecone
    image_data = []
    for idx, image_embedding in enumerate(image_embeddings):
        image_data.append({
            "id": str(image_filenames[idx]),  # unique ID for each image
            "values": image_embedding.tolist(),  # convert tensor to list for storing
            "metadata": {"filename": image_filenames[idx]}  # optional metadata
        })
        # When we have BATCH_SIZE items, upsert them
        if len(image_data) == BATCH_SIZE:
            index.upsert(vectors=image_data)
            image_data = []  # Clear the list after upserting
    
    logger.info("Desribe index stats with batching:")
    logger.info(index.describe_index_stats())
    # Upsert the embeddings into the Pinecone index
    if image_data:
        index.upsert(vectors=image_data)
    logger.info("Desribe index stats finally:")
    logger.info(index.describe_index_stats())
    
    
if __name__ == "__main__":
    pinecone = Pinecone(api_key="f6f8a0bb-375e-4a58-958c-c5469329ad9e")
    init_pinecone(pinecone)
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, paths.MODELS_PATH / "clip_model.pt")
    store_image_embeddings(pinecone, valid_df, image_embeddings, valid_df['image'].values)