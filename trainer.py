import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as func

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        super().__init__()

    @staticmethod
    def _batchify(x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for idx, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = func.binary_cross_entropy(y_hat_i, y_i)

            # Initialize gradient
            self.optimizer.zero_grad()
            loss_i.backward()

            # Gradient descent one step
            self.optimizer.step()

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        self.model.eval()

        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for idx, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = func.binary_cross_entropy(y_hat_i, y_i)

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config, to_be_shown=False):
        lowest_loss = np.inf
        lowest_epoch = np.inf
        best_model = None
        train_history, valid_history = [], []

        for epoch_idx in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            train_history += [train_loss]
            valid_history += [valid_loss]

            if (epoch_idx + 1) % config.print_interval == 0:
                print(f"Epoch({epoch_idx + 1}/{config.n_epochs}): train_loss={train_loss:.4e} valid_loss={valid_loss:.4e} lowest_loss={lowest_loss:.4e}")

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                lowest_epoch = epoch_idx
                best_model = deepcopy(self.model.state_dict())
            else:
                if config.early_stop > 0 and lowest_epoch + config.early_stop < epoch_idx + 1:
                    print(f"There is no improvement during last {config.early_stop} epochs.")
                    break

        if to_be_shown:
            import matplotlib.pyplot as plt

            plot_from = 2
            plt.figure(figsize=(20, 10))
            plt.grid(True)
            plt.title("Train / Valid Loss History")
            plt.plot(
                range(plot_from, len(train_history)), train_history[plot_from:],
                range(plot_from, len(valid_history)), valid_history[plot_from:],
            )
            plt.yscale('log')
            plt.show()

        print(f"The best validation loss for epoch {lowest_epoch + 1}: {lowest_loss:.4e}")

        # Restore best model
        self.model.load_state_dict(best_model)