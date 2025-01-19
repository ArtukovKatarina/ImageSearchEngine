import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer
from sentence_transformers import SentenceTransformer, util
from config import settings
from src import paths
from src.model.clip import Clip
from tqdm.autonotebook import tqdm
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Define evaluation metrics
def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant)
    return len(set(retrieved_at_k) & relevant_at_k) / k

def recall_at_k(retrieved, relevant, k):
    relevant_at_k = set(relevant)
    return len(set(retrieved[:k]) & relevant_at_k) / len(relevant_at_k)

def mean_average_precision(retrieved, relevant):
    relevant_set = set(relevant)
    ap_sum = 0.0
    relevant_count = 0

    for i, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            relevant_count += 1
            ap_sum += relevant_count / i

    return ap_sum / len(relevant) if relevant else 0

def ndcg_at_k(retrieved, relevant, k):
    def dcg(scores):
        return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))

    retrieved_at_k = retrieved[:k]
    ideal_at_k = sorted([1 if item in relevant else 0 for item in retrieved], reverse=True)[:k]

    actual_dcg = dcg([1 if item in relevant else 0 for item in retrieved_at_k])
    ideal_dcg = dcg(ideal_at_k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# Plot evaluation metrics
def plot_metrics(metrics):
    metrics_names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_names, values, color='skyblue')
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.xlabel("Metrics")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
    plt.show()

# Precompute caption embeddings
def precompute_caption_embeddings(valid_df, sentence_model):
    logger.info("Precomputing caption embeddings...")
    valid_df["caption_embedding"] = list(
        sentence_model.encode(valid_df["caption"].tolist(), convert_to_tensor=True)
    )
    logger.info("Finished precomputing caption embeddings.")
    return valid_df

# Visualize query results
def visualize_queries_results(queries_results):
    for query, results in queries_results.items():
        print(f"Query: {query}")
        print("Results:")
        for i, result in enumerate(results):
            print(f"  {i + 1}: {result['filename']} (score: {result['score']:.4f})")
        print("-" * 50)
        
# Get relevant images using semantic similarity
def get_relevant_images(query_caption, valid_df, sentence_model, top_n=5):
    query_embedding = sentence_model.encode(query_caption, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, torch.stack(valid_df["caption_embedding"].tolist()))[0]
    top_indices = similarities.argsort(descending=True)[:top_n]
    return valid_df.iloc[top_indices.cpu().numpy()]['image'].tolist()

# Process a single query
def process_query(row, valid_df, model, tokenizer, index, sentence_model, top_k, top_n_relevance):
    query_caption = row["caption"]
    ground_truth_image = row["image"]

    # Get relevant images
    relevant_images = get_relevant_images(query_caption, valid_df, sentence_model, top_n=top_n_relevance)

    # Tokenize caption
    encoded_query = tokenizer([query_caption], return_tensors="pt", padding=True)
    encoded_query = {k: v.to(paths.DEVICE) for k, v in encoded_query.items()}

    # Get text embedding
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=encoded_query["input_ids"], attention_mask=encoded_query["attention_mask"]
        )
        text_features = text_features.last_hidden_state[:, model.target_token_idx, :]
        text_embeddings = model.text_projection(text_features)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

    # Query Pinecone
    result = index.query(vector=text_embeddings[0].tolist(), top_k=top_k, include_metadata=True)
    retrieved_images = [match['metadata']['filename'] for match in result['matches']]

    # Calculate metrics
    metrics = {
        "precision_at_k": precision_at_k(retrieved_images, relevant_images, k=top_k),
        "recall_at_k": recall_at_k(retrieved_images, relevant_images, k=top_k),
        "map": mean_average_precision(retrieved_images, relevant_images),
        "ndcg_at_k": ndcg_at_k(retrieved_images, relevant_images, k=top_k)
    }
    
    return metrics, {query_caption: retrieved_images}

# Main evaluation function
def evaluate_image_search(valid_df, model_path, pinecone_index_name, sentence_model, top_k=10, top_n_relevance=5):
    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)

    # Load model
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

    # Initialize Pinecone
    pinecone = Pinecone(api_key=settings.api_key)
    index = pinecone.Index(pinecone_index_name)

    # Precompute caption embeddings
    valid_df = precompute_caption_embeddings(valid_df, sentence_model)

    # Parallel processing of queries
    metrics_list = []
    queries_results = {}

    logger.info("Starting evaluation...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_query, row, valid_df, model, tokenizer, index, sentence_model, top_k, top_n_relevance)
            for _, row in valid_df.iterrows()
        ]

        for future in tqdm(futures):
            metrics, query_result = future.result()
            metrics_list.append(metrics)
            queries_results.update(query_result)

    # Aggregate metrics
    aggregated_metrics = {
        "precision_at_k": np.mean([m["precision_at_k"] for m in metrics_list]),
        "recall_at_k": np.mean([m["recall_at_k"] for m in metrics_list]),
        "map": np.mean([m["map"] for m in metrics_list]),
        "ndcg_at_k": np.mean([m["ndcg_at_k"] for m in metrics_list])
    }

    logger.info(f"Aggregated Metrics: {aggregated_metrics}")
    return aggregated_metrics, queries_results

if __name__ == "__main__":
    # Load validation dataframe
    from src.train.train import make_train_valid_dfs
    train_df, valid_df = make_train_valid_dfs()

    # Load pre-trained sentence transformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Evaluate the model
    metrics, queries_results = evaluate_image_search(
        valid_df=valid_df,
        model_path=paths.MODELS_PATH / "clip_model.pt",
        pinecone_index_name=settings.pinecone_index_name,
        sentence_model=sentence_model,
        top_k=10,
        top_n_relevance=5
    )

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualize metrics
    plot_metrics(metrics)

    # Visualize query results
    visualize_queries_results(queries_results)