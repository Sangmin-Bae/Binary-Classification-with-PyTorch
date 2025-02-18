import pandas as pd

import torch

def load_data(is_full=False):
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data['class'] = cancer.target

    if not is_full:
        # Select More Valid Features
        cols = ["mean radius", "mean texture",
                "mean smoothness", "mean compactness", "mean concave points",
                "worst radius", "worst texture",
                "worst smoothness", "worst compactness", "worst concave points",
                "class"]

        data = torch.from_numpy(data[cols].values).float()

    else:
        data = torch.from_numpy(data.values).float()

    x, y = data[:, :-1], data[:, -1:]

    return x, y