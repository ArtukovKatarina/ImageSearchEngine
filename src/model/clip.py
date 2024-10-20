import torch
from torch import nn
import torch.nn.functional as F
import timm
from src import paths
from config import settings
from transformers import DistilBertConfig, DistilBertModel

class Clip(nn.Module):
    def __init__(
        self, 
        temperature, 
        image_embedding,
        text_embedding,
        model_name,
        text_encoder_model,
        pretrained,
        trainable,
        projection_dim,
        dropout
    ):
        super().__init__()
        
        # Image Encoder
        self.image_encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.image_encoder.parameters():
            p.requires_grad = trainable

        # Text Encoder
        if pretrained:
            self.text_encoder = DistilBertModel.from_pretrained(text_encoder_model)
        else:
            self.text_encoder = DistilBertModel(config=DistilBertConfig())

        for p in self.text_encoder.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        # Projection Heads
        self.image_projection = nn.Linear(image_embedding, projection_dim)
        self.text_projection = nn.Linear(text_embedding, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.temperature = temperature

    def forward(self, batch):
        # Image Encoding
        image_features = self.image_encoder(batch["image"])
        # Text Encoding
        text_output = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_features = text_output.last_hidden_state[:, self.target_token_idx, :]

        # Image Projection
        image_embeddings = self.projection(image_features, self.image_projection)
        # Text Projection
        text_embeddings = self.projection(text_features, self.text_projection)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def projection(self, x, projection_layer):
        projected = projection_layer(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
