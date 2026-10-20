"""Trains the DoA-conditioned HRTF prior. Usage: uv run python train.py dataset_dir=<pointwise_hrir_44p1kHz/hutubs_simulated>"""
import os
from itertools import cycle

import hydra
import torch

from training.trainer import Trainer


@hydra.main(config_path="training", config_name="train_score_model", version_base=str(hydra.__version__))
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model_dir = hydra.utils.to_absolute_path(args.model_dir)
    os.makedirs(args.model_dir, exist_ok=True)

    train_loader = cycle(iter(hydra.utils.instantiate(args.train_set_dataloader)))
    network = hydra.utils.instantiate(args.network).to(device)
    diff_params = hydra.utils.instantiate(args.diff_params)

    Trainer(args=args, dset=train_loader, network=network, diff_params=diff_params, device=device).training_loop()


if __name__ == "__main__":
    main()
