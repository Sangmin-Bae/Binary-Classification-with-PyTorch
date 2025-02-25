import torch
import torch.nn.functional as func

from model.fc_model import FullyConnectedClassifier

from utils import load_data
from utils import split_data
from utils import preprocessing_data

def load_model(model_fn, device):
    d = torch.load(model_fn, map_location=device, weights_only=False)

    return d["model"], d["config"]

def test(model, x, y, batch_size, to_be_shown=False):
    model.eval()

    test_loss = 0
    y_hat = []

    with torch.no_grad():
        x_ = x.split(batch_size, dim=0)
        y_ = y.split(batch_size, dim=0)

        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = func.binary_cross_entropy(y_hat_i, y_i)

            test_loss += loss

            y_hat += [y_hat_i]

        test_loss = test_loss / len(x_)
        y_hat = torch.cat(y_hat, dim=0)

        print(f"Test loss: {test_loss:.4e}")

        correct_cnt = (y == (y_hat > .5)).sum()
        total_cnt = float(y.size(0))

        print(f"Accuracy : {correct_cnt/total_cnt:.4f}")

        if to_be_shown:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_auc_score

            y, y_hat = y.to("cpu"), y_hat.to("cpu")

            df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(), columns=['y', "y_hat"])

            roc_auc_score = roc_auc_score(df.values[:, 0], df.values[:, 1])
            print(f"ROC AUC Score: {roc_auc_score}")

            sns.histplot(df, x="y_hat", hue='y', bins=50, stat="probability")
            plt.show()

def main():
    model_fn = "./models/fc_model.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dict, config = load_model(model_fn, device)

    # Load data
    x, y = load_data()
    x, y = split_data(x, y, device, config.train_ratio)
    x, y = preprocessing_data(x, y, device, is_train=False)

    model = FullyConnectedClassifier(input_size=x.size(-1), output_size=y.size(-1)).to(device)

    model.load_state_dict(model_dict)

    test(model, x, y, config.batch_size, to_be_shown=True)

if __name__ == "__main__":
    main()