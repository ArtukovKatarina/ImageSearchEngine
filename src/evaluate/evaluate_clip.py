import torch
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from config import settings
from src import paths
from src.model.clip import Clip
from transformers import DistilBertTokenizer
import cv2  # For image loading

logger = logging.getLogger(__name__)

def calculate_top_k_accuracy(similarities, target_index, k):
    """
    Calculate Top-K accuracy.
    """
    max_k = min(k, similarities.size(-1))  # Ensure k does not exceed available samples
    top_k_indices = similarities.topk(max_k, dim=-1).indices
    return int(target_index in top_k_indices)

def evaluate_clip_top_k(model, tokenizer, valid_df, device, top_k_values=[1, 3, 5, 10]):
    """
    Evaluate the CLIP model for Top-K accuracy.
    """
    results = {f"top-{k}": 0 for k in top_k_values}
    total_samples = len(valid_df)

    with torch.no_grad():
        for _, row in tqdm(valid_df.iterrows(), total=total_samples):
            # Process text (caption)
            caption = row["caption"]
            text_inputs = tokenizer([caption], return_tensors="pt", padding=True, truncation=True).to(device)

            # Process image
            image_path = paths.RAW_PATH / row["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Image at {image_path} could not be loaded. Skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (settings.image_size, settings.image_size))
            image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
            image /= 255.0  # Normalize image pixel values to [0, 1]

            # Get embeddings
            text_features = F.normalize(
                model.text_projection(
                    model.text_encoder(
                        input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"]
                    ).last_hidden_state[:, model.target_token_idx, :]
                ), dim=-1
            )
            image_features = F.normalize(model.image_projection(model.image_encoder(image)), dim=-1)

            # Compute similarities
            similarities = 100.0 * (image_features @ text_features.T)

            # Calculate Top-K accuracies
            target_index = row["id"]  # Assuming `id` column represents the ground truth index
            for k in top_k_values:
                results[f"top-{k}"] += calculate_top_k_accuracy(similarities, target_index, k)

    # Normalize results to get accuracies
    for k in top_k_values:
        results[f"top-{k}"] /= total_samples

    return results

def main():
    logger.info("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model
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
    ).to(device)
    model.load_state_dict(torch.load(paths.MODELS_PATH / "clip_model.pt", map_location=device))
    model.eval()

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)

    # Load dataset
    from src.train.train import make_train_valid_dfs
    train_df, valid_df = make_train_valid_dfs()

    # Evaluate model
    logger.info("Starting evaluation...")
    top_k_values = [1, 3, 5, 10]
    results = evaluate_clip_top_k(model, tokenizer, valid_df, device, top_k_values)

    # Log results
    logger.info("Top-K Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
