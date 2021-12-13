from omegaconf import DictConfig
import hydra
from dataset import build_loaders
from models import CLAP
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from utils import AvgMeter
import torchaudio


@hydra.main(config_path="configs")
def train(cfg: DictConfig) -> None:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    train_dataloader = build_loaders(cfg=cfg.pretrain, split=cfg.pretrain.train_split)
    valid_dataloader = build_loaders(cfg=cfg.pretrain, split=cfg.pretrain.valid_split)

    model = CLAP(text_cfg=cfg.model.text, audio_cfg=cfg.model.audio, embed_dim=cfg.pretrain.embed_dim).to(device)

    # todo: fine tune lr
    # params = {
    #
    # }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=cfg.pretrain.patience,
        factor=cfg.pretrain.factor,
    )

    best_loss = float("inf")

    for epoch in range(cfg.pretrain.epochs):
        print(f"Epoch: {epoch+1}")
        model.train()
        train_epoch(
            model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=cfg.pretrain.batch_size,
        )
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(
                model=model,
                valid_loader=valid_dataloader,
                device=device,
                batch_size=cfg.pretrain.batch_size,
            )

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), hydra.utils.get_original_cwd()+"/"+cfg.pretrain.cp_path)
            print("Saved Best Model")

        lr_scheduler.step(valid_loss)


def train_epoch(model, train_loader, optimizer, device, batch_size) -> None:
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    loss_text = nn.CrossEntropyLoss(reduction="mean")
    loss_audio = nn.CrossEntropyLoss(reduction="mean")

    for batch in tqdm_object:
        optimizer.zero_grad()

        mfcc = batch['mfcc'].to(device)
        text = batch['text'].to(device)

        logits_per_audio, logits_per_text = model(mfcc, text)

        ground_truth = torch.arange(batch_size, device=device)

        loss = (loss_audio(logits_per_audio, ground_truth) + loss_text(logits_per_text, ground_truth))/2
        loss.backward()

        optimizer.step()

        wandb.log({'loss': loss.item()})

        loss_meter.update(loss.item(), count=batch_size)
        tqdm_object.set_postfix(train_loss=loss_meter.avg)


def valid_epoch(model, valid_loader, device, batch_size):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    loss_audio = nn.CrossEntropyLoss(reduction="mean")
    loss_text = nn.CrossEntropyLoss(reduction="mean")

    for batch in tqdm_object:
        mfcc = batch['mfcc'].to(device)
        text = batch['text'].to(device)

        logits_per_audio, logits_per_text = model(mfcc, text)

        ground_truth = torch.arange(batch_size, device=device)

        loss = (loss_audio(logits_per_audio, ground_truth) + loss_text(logits_per_text, ground_truth)) / 2

        loss_meter.update(loss.item(), count=batch_size)
        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    return loss_meter.avg


if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(project="clap-pretrain")
    train()
