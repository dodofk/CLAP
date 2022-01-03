from omegaconf import DictConfig
import hydra
from dataset import build_loaders
from models import CLAP
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from utils import AvgMeter
from transformers import AutoTokenizer


@hydra.main(config_path="configs")
def train(cfg: DictConfig) -> None:
    device = cfg.pretrain.device if torch.cuda.is_available() else "cpu"

    train_dataloader = build_loaders(cfg=cfg.pretrain, split=cfg.pretrain.train_split)
    valid_dataloader = build_loaders(cfg=cfg.pretrain, split=cfg.pretrain.valid_split)

    audio_feature = cfg.model.audio.feature

    model = CLAP(text_cfg=cfg.model.text, audio_cfg=cfg.model.audio, embed_dim=cfg.pretrain.embed_dim).to(device)

    # todo: fine tune lr
    # params = {
    #
    # }

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay,
        betas=(cfg.pretrain.beta_1, cfg.pretrain.beta_2),
        eps=cfg.pretrain.eps,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=5,
        eta_min=5e-6,
    )

    best_loss = float("inf")

    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrain.tokenizer) if cfg.model.text.name in ['distilbert'] else None

    for epoch in range(cfg.pretrain.epochs):
        print(f"Epoch: {epoch+1}")
        model.train()
        train_epoch(
            model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            batch_size=cfg.pretrain.batch_size,
            audio_feature=audio_feature,
            tokenizer=tokenizer,
        )
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(
                model=model,
                valid_loader=valid_dataloader,
                device=device,
                batch_size=cfg.pretrain.batch_size,
                audio_feature=audio_feature,
                tokenizer=tokenizer,
            )

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), hydra.utils.get_original_cwd()+"/"+cfg.pretrain.cp_path)
            print("Saved Best Model")

        lr_scheduler.step()


def train_epoch(model, train_loader, optimizer, device, batch_size, audio_feature, tokenizer=None) -> None:
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    loss_text = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_audio = nn.CrossEntropyLoss(label_smoothing=0.1)

    for batch in tqdm_object:
        optimizer.zero_grad()

        audio = batch[audio_feature].to(device)
        if tokenizer is not None:
            text = tokenizer(
                batch['transcript'],
                return_tensors='pt',
                padding=True,
            ).to(device)
        else:
            text = batch['text'].to(device)

        logits_per_audio, logits_per_text = model(audio, text)

        ground_truth = torch.arange(batch_size, device=device)

        audio_loss = loss_audio(logits_per_audio, ground_truth)

        text_loss = loss_text(logits_per_text, ground_truth)

        loss = (audio_loss + text_loss)/2
        loss.backward()

        optimizer.step()

        wandb.log({'loss': loss.item(), 'audio_loss': audio_loss.item(), 'text_loss': text_loss.item()})

        loss_meter.update(loss.item(), count=batch_size)
        tqdm_object.set_postfix(train_loss=loss_meter.avg)


def valid_epoch(model, valid_loader, device, batch_size, audio_feature, tokenizer=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    loss_audio = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_text = nn.CrossEntropyLoss(label_smoothing=0.1)

    for batch in tqdm_object:
        audio = batch[audio_feature].to(device)
        if tokenizer is not None:
            text = tokenizer(
                batch['transcript'],
                return_tensors='pt',
                padding=True,
            ).to(device)
        else:
            text = batch['text'].to(device)


        logits_per_audio, logits_per_text = model(audio, text)

        ground_truth = torch.arange(batch_size, device=device)

        loss = (loss_audio(logits_per_audio, ground_truth) + loss_text(logits_per_text, ground_truth)) / 2

        loss_meter.update(loss.item(), count=batch_size)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        wandb.log({'valid_loss': loss_meter.avg})

    return loss_meter.avg


if __name__ == "__main__":
    torch.cuda.empty_cache()
    wandb.init(project="clap-pretrain")
    train()
