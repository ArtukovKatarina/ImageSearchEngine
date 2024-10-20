import torch
import logging
import pandas as pd
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
    INDEX_NAME = settings.pinecone_index_name

    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(INDEX_NAME)
    pinecone.create_index(name=INDEX_NAME, 
        dimension=settings.projection_dim, 
        metric= settings.pinecone_metric,
        spec=ServerlessSpec(cloud= settings.pinecone_cloud, region=settings.pinecone_region))

    pinecone.Index(INDEX_NAME)
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
    logger.info("Image filenames number: %f", len(image_filenames))
    
    index = pinecone.Index(settings.pinecone_index_name)
    
    batch_size = settings.pinecone_batch_size
    num_embeddings = len(image_embeddings)
    
    logger.info("Starting the upsert process in batches of size: %d", batch_size)
    logger.info(f"Total number of embeddings: {len(image_embeddings)}")
    
    for i in range(0, num_embeddings, batch_size):
        batch_embeddings = image_embeddings[i:i + batch_size]
        batch_filenames = image_filenames[i:i + batch_size]
        
        image_data = []
        for idx, image_embedding in enumerate(batch_embeddings):
            image_data.append({
                "id": str(batch_filenames[idx]), 
                "values": image_embedding.tolist(), 
                "metadata": {"filename": batch_filenames[idx]} 
            })
            
        df = pd.DataFrame(image_data)
        logger.info(f"Upserting batch {i // batch_size + 1}/{num_embeddings // batch_size + 1}")
        index.upsert_from_dataframe(df, batch_size = settings.pinecone_batch_size, show_progress=True)
    
    logger.info("Desribe index stats after upserts:")
    logger.info(index.describe_index_stats())
    
if __name__ == "__main__":
    pinecone = Pinecone(api_key=settings.api_key)
    init_pinecone(pinecone)
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, paths.MODELS_PATH / "clip_model.pt")
    store_image_embeddings(pinecone, valid_df, image_embeddings, valid_df['image'].values)