import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import logging
from transformers import DistilBertTokenizer
from config import settings
from src import paths
from src.model.clip import Clip
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

app = FastAPI()


def find_matches_in_vector_db(pinecone, model, query, n=9):
    logger.info("Start find matches method")

    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(paths.DEVICE)
        for key, values in encoded_query.items()
    }
    
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_features = text_features.last_hidden_state[:, model.target_token_idx, :]  # CLS token !?!?!?
        text_embeddings = model.text_projection(text_features)  # Project to 256-dim
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)


    logger.info("Created embedding from query")
    # Query Pinecone for the top N similar images
    index = pinecone.Index("clip-image-text-search")
    result = index.query(vector=text_embeddings[0].tolist(), top_k=n, include_metadata=True)

    # Retrieve the filenames of the matching images
    matches = [match['metadata']['filename'] for match in result['matches']]

    logger.info(matches)
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches[:9], axes.flatten()):
        logger.info(match)
        image_path = f"{paths.RAW_PATH}/{match}"
        logger.info(image_path)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.axis("off")
        else:
            logger.warning(f"Warning: Image at {image_path} could not be loaded.")
    
    plt.show()
    
if __name__ == "__main__":
    pinecone = Pinecone(api_key="f6f8a0bb-375e-4a58-958c-c5469329ad9e")
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
    model.load_state_dict(torch.load(paths.MODELS_PATH / "clip_model.pt", map_location=paths.DEVICE))
    model.eval()
    find_matches_in_vector_db(pinecone, 
                              model,
                              query="man working in field",
                              n=9)