import torch

from model import MyModel

from utils import load_data

def load_model(model_fn, device):
    d = torch.load(model_fn, map_location=device, weights_only=False)

    return d["model"], d["config"]

def test(model, x, y, to_be_shown=False):
    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y == (y_hat > .5)).sum()
        total_cnt = float(y.size(0))

        print(f"Accuracy : {correct_cnt/total_cnt:.4f}")

        if to_be_shown:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            y, y_hat = y.to("cpu"), y_hat.to("cpu")

            df = pd.DataFrame(torch.cat([y, y_hat], dim=1).detach_().numpy(), columns=['y', "y_hat"])
            sns.histplot(df, x="y_hat", hue='y', bins=50, stat="probability")
            plt.show()

def main():
    model_fn = "./model/model.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    x, y = load_data(is_full=False)

    model_dict, config = load_model(model_fn, device)

    model = MyModel(input_size=x.size(-1), output_size=y.size(-1)).to(device)

    model.load_state_dict(model_dict)

    test(model, x.to(device), y.to(device), to_be_shown=True)

if __name__ == "__main__":
    main()