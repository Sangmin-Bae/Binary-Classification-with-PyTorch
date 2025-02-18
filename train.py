import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import MyModel

from utils import load_data

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="model_file_name")
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument("--n_epochs", type=int, default=200000, help="number_of_epochs")
    p.add_argument("--lr", type=float, default=1e-2, help="learning_rate")

    config = p.parse_args()

    return config

def main(config):
    # Device Check
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    print(f"Device : {device}")

    # Load Data
    x, y = load_data(is_full=False)
    print(f"Train Data Shape : {x.shape}")
    print(f"Target Data Shape : {y.shape}")

    # Define Model
    model = MyModel(input_size=x.size(-1), output_size=y.size(-1)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    crit = nn.BCELoss()

    print(f"Model : {model}")
    print(f"Optimizer : {optimizer}")
    print(f"crit : {crit}")

    # Train

    # Save Model
    pass

if __name__ == "__main__":
    config = argument_parser()
    main(config)
