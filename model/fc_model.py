import torch
import torch.nn as nn


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, 25),
            nn.LeakyReLU(),
            nn.Linear(25, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Linear(5, output_size),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.act(self.layer(x))
        # |y| = (batch_size, output_size)

        return y