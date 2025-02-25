import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from model.fc_model import FullyConnectedClassifier
from trainer import Trainer

from utils import load_data
from utils import split_data
from utils import preprocessing_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, required=True, help="config_file_path")

    args = p.parse_args()

    return args

def load_config(path):
    with open(path) as f:
        config = argparse.Namespace(**yaml.safe_load(f))

    return config

def main(config):
    # Device Check
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data()
    x, y = split_data(x, y, device, config.train_ratio)
    x, y = preprocessing_data(x, y, device, is_train=True)

    print(f"Train: {x[0].shape} {y[0].shape}")
    print(f"Valid: {x[1].shape} {y[1].shape}")

    # Define Model
    model = FullyConnectedClassifier(input_size=x[0].size(-1), output_size=y[0].size(-1)).to(device)
    optimizer = optim.Adam(model.parameters())

    print(f"Model : {model}")
    print(f"Optimizer : {optimizer}")

    # Train
    trainer = Trainer(model, optimizer)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config,
        to_be_shown=False
    )

    # Save best model weights
    torch.save({
        "model": trainer.model.state_dict(),
        "opt": optimizer.state_dict(),
        "config": config
    }, config.model_fn)

if __name__ == "__main__":
    args = argument_parser()
    config = load_config(args.config)
    main(config)
