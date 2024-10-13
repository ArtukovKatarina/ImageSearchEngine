import torch
import cv2
import numpy as np
import torch.nn.functional as F
import logging
from transformers import DistilBertTokenizer
from config import settings
from src import paths
from src.model.clip import Clip
from pinecone import Pinecone
from fastapi import FastAPI
import base64
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.data import loader

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pinecone = Pinecone(api_key=settings.api_key)
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

class SearchRequest(BaseModel):
    query: str
    n: int = 9

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def decode_base64_to_image(base64_string: str):
    img_data = base64.b64decode(base64_string)
    np_img = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    transforms = loader.get_transforms(settings.image_size, mode="valid")
    image = transforms(image=image)["image"]
    image_tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0).to(paths.DEVICE)
    
    return image_tensor

@app.post("/search_by_text")
def search_vector_db_by_text(search_request : SearchRequest):
    logger.info("Start find matches method")
    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)
    encoded_query = tokenizer([search_request.query])
    batch = {
        key: torch.tensor(values).to(paths.DEVICE)
        for key, values in encoded_query.items()
    }
    
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_features = text_features.last_hidden_state[:, model.target_token_idx, :]  # CLS token
        text_embeddings = model.text_projection(text_features)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    logger.info("Created embedding from query")
    
    index = pinecone.Index("clip-image-text-search")
    result = index.query(vector=text_embeddings[0].tolist(), top_k=search_request.n, include_metadata=True)

    matches = [match['metadata']['filename'] for match in result['matches']]
    logger.info(matches)
    
    image_data_list = []
    for match in matches:
        image_path = f"{paths.RAW_PATH}/{match}"
        logger.info(f"Image path: {image_path}")
        
        if cv2.imread(image_path) is not None:
            encoded_image = encode_image_to_base64(image_path)
            image_data_list.append(f"data:image/jpeg;base64,{encoded_image}")
        else:
            logger.warning(f"Warning: Image at {image_path} could not be loaded.")
    
    return {"images": image_data_list}

@app.post("/search_by_image")
def search_vector_db_by_image(search_request: SearchRequest):
    logger.info("Start find matches method by image")
    
    try:
        decoded_image = decode_base64_to_image(search_request.query)
        logger.info("Decoded the image successfully")
    except Exception as e:
        logger.error(f"Failed to decode image: {str(e)}")
        return {"error": "Invalid image format or decoding failed."}
    
    preprocessed_image = preprocess_image(decoded_image)
    
    with torch.no_grad():
        image_features = model.image_encoder(preprocessed_image)
        image_features = model.image_projection(image_features)
        image_embeddings = F.normalize(image_features, p=2, dim=-1)
    logger.info("Created embedding from image")
    
    index = pinecone.Index("clip-image-text-search")
    result = index.query(vector=image_embeddings[0].tolist(), top_k=search_request.n, include_metadata=True)

    matches = [match['metadata']['filename'] for match in result['matches']]
    logger.info(matches)
    
    image_data_list = []
    for match in matches:
        image_path = f"{paths.RAW_PATH}/{match}"
        logger.info(f"Image path: {image_path}")
        
        if cv2.imread(image_path) is not None:
            encoded_image = encode_image_to_base64(image_path)
            image_data_list.append(f"data:image/jpeg;base64,{encoded_image}")
        else:
            logger.warning(f"Warning: Image at {image_path} could not be loaded.")
    
    return {"images": image_data_list}
