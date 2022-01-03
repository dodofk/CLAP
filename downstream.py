from omegaconf import DictConfig
import hydra
from models import E2ESLU
from dataset import build_loaders
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from utils import AvgMeter


@hydra.main(config_path="configs")
def train(cfg: DictConfig) -> None:
    device = cfg.downstream.device if torch.cuda.is_available() else "cpu"

    train_dataloader = build_loaders(cfg=cfg.downstream, split=cfg.downstream.train_split)
    valid_dataloader = build_loaders(cfg=cfg.downstream, split=cfg.downstream.valid_split)

    model = E2ESLU(
        text_config=cfg.model.text, 
        audio_cfg=cfg.model.audio, 
        check_point_path=cfg.downstream.cp_path, 
        embed_dim=cfg.downstream.embed_dim,
        hidden_dim=cfg.downstream.hidden_dim,
        output_dim=cfg.downstream.output_dim,
        trainable=cfg.downstream.trainable,
        from_pretrain=cfg.downstream.from_pretrain,
    ).to(device)

    params = [
        {"params": model.audio_encoder.parameters(), 'lr': cfg.downstream.audio_lr},
        {"params": model.final_classifier.parameters(), 'lr': cfg.downstream.classifier_lr},
    ]

    optimizer = torch.optim.AdamW(
        params=params,
        weight_decay=cfg.downstream.weight_decay,
    )

    best_loss = float("inf")

    for epoch in range(cfg.downstream.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_epoch(
            model=model, 
            train_loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            audio_feature=cfg.model.audio.feature, 
        )
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(
                model=model,
                valid_loader=valid_dataloader,
                device=device,
                audio_feature=cfg.model.audio.feature,
            )

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), hydra.utils.get_original_cwd()+"/"+cfg.downstream.finetune_cp_path)
            print("Saved Best Model!")


def train_epoch(model, train_loader, optimizer, device, audio_feature) -> None:
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    train_total = 0
    train_acc = 0

    for batch in tqdm_object:
        output = model(batch[audio_feature].to(device))
        target = batch['intent'].to(device)
        pred = torch.argmax(output, dim=1)

        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = batch[audio_feature].size(0)

        train_total += count
        train_acc += (pred.to("cpu") == target.to("cpu")).sum().item()

        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_acc=train_acc/train_total)

        wandb.log({'train_loss': loss.item(), 'train_acc': train_acc/train_total})


def valid_epoch(model, valid_loader, device, audio_feature):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    valid_total = 0
    valid_acc = 0

    for batch in tqdm_object:
        output = model(batch[audio_feature].to(device))
        target = batch['intent'].to(device)
        pred = torch.argmax(output, dim=1)

        loss = loss_fn(output, target)
        

        count = batch[audio_feature].size(0)

        valid_total += count
        valid_acc += (pred.to("cpu") == target.to("cpu")).sum().item()

        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg, valid_acc=valid_acc/valid_total)

        wandb.log({'valid_loss': loss.item(), 'valid_acc': valid_acc/valid_total})

    return loss_meter.avg


if __name__ == "__main__":
    wandb.init(project="clap-fsc-finetune")
    train()
