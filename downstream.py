from omegaconf import DictConfig
import hydra
from dataset import build_loaders


import torchaudio


@hydra.main(config_path="configs")
def train(cfg: DictConfig) -> None:
    train_dataloader = build_loaders(cfg=cfg.downstream, split=cfg.downstream.train_split)
    train_iter = iter(train_dataloader)
    batch = next(train_iter)


if __name__ == "__main__":
    train()
