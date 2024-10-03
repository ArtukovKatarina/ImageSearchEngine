import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import logging
import numpy as np
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer
from config import settings
from src import paths
from src.model.clip import Clip
from src.train.train import build_loaders, make_train_valid_dfs

logger = logging.getLogger(__name__)

def get_image_embeddings(valid_df, model_path):
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
    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=9):
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
        # Extract the tensor from the BaseModelOutput ... it must be tensor
        text_features = text_features.last_hidden_state  # or another appropriate attribute
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    # matches = [image_filenames[idx] for idx in indices[::5].tolist()] 
    seen_images = set()
    matches = []

    for idx in indices.tolist():
        match = image_filenames[idx]
        if isinstance(match, (list, np.ndarray)):
            # Iterate through each element in the list to find unique images
            for submatch in match:
                if submatch not in seen_images:
                    seen_images.add(submatch)
                    matches.append(submatch)
        else:
            if match not in seen_images:
                seen_images.add(match)
                matches.append(match)

        if len(matches) >= n:
            break
    
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
            print(f"Warning: Image at {image_path} could not be loaded.")
    
    plt.show()
    
if __name__ == "__main__":
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, paths.MODELS_PATH / "clip_model.pt")
    find_matches(model, 
             image_embeddings,
             query="a girl drinking wine",
             image_filenames=valid_df['image'].values,
             n=9)