import pandas as pd
import numpy as np
import itertools
import logging
import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer
from config import settings
from src import paths
from src.data import loader
from src.data.loader import CLIPLoader
from src.utils import avg_meter
from src.utils.avg_meter import AvgMeter
from src.model.clip import Clip

logger = logging.getLogger(__name__)

def make_train_valid_dfs():
    dataframe = pd.read_csv(paths.PROCESSED_PATH / "captions.csv")
    max_id = dataframe["id"].max() + 1 if not settings.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = loader.get_transforms(settings.image_size, mode=mode)
    dataset = CLIPLoader(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        max_length = settings.max_length
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(paths.DEVICE) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=avg_meter.get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(paths.DEVICE) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def plot_graphs(history, metric):
    plt.plot(history[metric])
    plt.plot(history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])

def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(settings.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
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
    params = [
        {"params": model.image_encoder.parameters(), "lr": settings.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": settings.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": settings.head_lr, "weight_decay": settings.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=settings.patience, factor=settings.factor
    )
    step = "epoch"

    best_loss = float('inf')
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    for epoch in range(settings.epochs):
        logger.info("Epoch: %i", epoch +1)
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        history['loss'].append(train_loss.avg)
        history['val_loss'].append(valid_loss.avg)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            logger.info("Saving model...")
            torch.save(model.state_dict(), paths.MODELS_PATH / "clip_model.pt")
            logger.info("Saved Clip Model!")
        
        lr_scheduler.step(valid_loss.avg)
        
    logger.info("Train loss: %f", train_loss.avg)
    logger.info("Valid loss: %f", valid_loss.avg)

    # Plotting loss graphs
    plot_graphs(history, 'loss')
    plt.show()
        
if __name__ == "__main__":
    main()