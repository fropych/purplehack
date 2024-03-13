from datetime import datetime

import hydra
import numpy as np
import torch
from model import SwapNoiseMasker, TransformerAutoEncoder
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dae.data import SingleDataset, get_data


class AverageMeter(object):
    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val: float, n: int = 1) -> None:
        """Update the average meter with a new value and its frequency.

        Args:
            val (float): The new value to be added to the meter.
            n (int): The frequency of the new value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_dae(
    model: TransformerAutoEncoder,
    noise_maker: SwapNoiseMasker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dl: torch.utils.data.DataLoader,
    max_epochs: int,
    model_checkpoint: str = "model_checkpoint.pth",
) -> None:
    """
    Train the denoising autoencoder.

    Args:
        model: The autoencoder model.
        noise_maker: The noise maker object.
        optimizer: The optimizer object.
        scheduler: The learning rate scheduler object.
        train_dl: The training data loader.
        max_epochs: The maximum number of epochs.
        model_checkpoint: The path to save the trained model.

    Returns:
        None
    """
    for epoch in range(max_epochs):
        t0 = datetime.now()
        model.train()
        meter = AverageMeter()

        for i, x in enumerate(tqdm(train_dl)):
            x = x.cuda()
            x_corrputed, mask = noise_maker.apply(x)
            optimizer.zero_grad()
            loss = model.loss(x_corrputed, x, mask)
            loss.backward()
            optimizer.step()

            meter.update(loss.detach().cpu().numpy())

        delta = (datetime.now() - t0).seconds
        scheduler.step()
        print(
            f"\r epoch {epoch:5d} - loss {meter.avg:.6f} - {delta:4.6f} sec per epoch",
        )

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
        },
        model_checkpoint,
    )


def extract_features(
    model: torch.nn.Module, dataset: torch.utils.data.Dataset, batch_size: int
) -> torch.Tensor:
    """Extract features from a model for a given dataset using batch processing.

    Args:
        model (nn.Module): The dae model to extract features from.
        dataset (torch.utils.data.Dataset): The dataset to extract features from.
        batch_size (int): The number of samples to process in parallel.

    Returns:
        torch.Tensor: A tensor containing the extracted features.
    """
    dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    features = []
    model.eval()
    with torch.no_grad():
        for x in dl:
            features.append(model.feature(x.cuda()).detach().cpu())
    features = torch.vstack(features)
    return features


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Train and extract features from a denoising autoencoder.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        None
    """
    X, n_cats, n_nums = get_data(
        "data/train_data.pqt",
        "data/train_data.pqt",
        [
            "ogrn_month",
        ],
    )

    repeats = [n_cats, n_nums]
    probas = cfg.probas
    swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])

    dataset = SingleDataset(X)
    train_dl = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    model = TransformerAutoEncoder(
        num_inputs=X.shape[1], n_cats=n_cats, n_nums=n_nums, **cfg.model_params
    ).cuda()
    noise_maker = SwapNoiseMasker(swap_probas)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)

    train_dae(model, noise_maker, optimizer, scheduler, train_dl, cfg.max_epochs)

    features = extract_features(model, dataset, cfg.batch_size)
    np.save("dae_features.npy", features.numpy())


if __name__ == "__main__":
    main()
