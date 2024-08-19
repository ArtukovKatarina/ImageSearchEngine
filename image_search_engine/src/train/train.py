import pandas as pd
import numpy as np
import itertools
import torch
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer
from image_search_engine.src import config as CFG
from image_search_engine.src.data import loader
from image_search_engine.src.data.loader import CLIPLoader
from image_search_engine.src.utils import avg_meter
from image_search_engine.src.utils.avg_meter import AvgMeter
from image_search_engine.src.model.clip import Clip

def make_train_valid_dfs():
    dataframe = pd.read_csv(CFG.PROCESSED_PATH / "captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.DEBUG else 100
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
    transforms = loader.get_transforms(mode=mode)
    dataset = CLIPLoader(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.DEVICE) for k, v in batch.items() if k != "caption"}
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
        batch = {k: v.to(CFG.DEVICE) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.TEXT_TOKENIZER)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = Clip().to(CFG.DEVICE)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.IMAGE_ENCODER_LR},
        {"params": model.text_encoder.parameters(), "lr": CFG.TEXT_ENCODER_LR},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.HEAD_LR, "weight_decay": CFG.WEIGHT_DECAY}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.PATIENCE, factor=CFG.FACTOR
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.EPOCHS):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
        
if __name__ == "__main__":
    main()