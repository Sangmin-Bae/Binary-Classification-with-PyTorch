import pandas as pd

import torch

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_data():
    cancer = load_breast_cancer()
    data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data['class'] = cancer.target

    data = torch.from_numpy(data.values).float()

    x, y = data[:, :-1], data[:, -1:]

    return x, y


def split_data(x, y, device, ratios=(.6, .2, .2)):
    x = x.to(device)
    y = y.to(device)

    train_cnt = int(x.size(0) * ratios[0])
    valid_cnt = int(x.size(0) * ratios[1])
    test_cnt = x.size(0) - train_cnt - valid_cnt
    cnt = [train_cnt, valid_cnt, test_cnt]

    indices = torch.randperm(x.size(0)).to(device)
    x = torch.index_select(x, dim=0, index=indices)
    y = torch.index_select(y, dim=0, index=indices)

    x = x.split(cnt, dim=0)
    y = y.split(cnt, dim=0)

    return x, y


def preprocessing_data(x, y, device, is_train=True):
    scaler = StandardScaler()
    scaler.fit(x[0].to('cpu').numpy())

    preprocessed_x = [
        torch.from_numpy(scaler.transform(x[0].to('cpu').numpy())).float(),
        torch.from_numpy(scaler.transform(x[1].to('cpu').numpy())).float(),
        torch.from_numpy(scaler.transform(x[2].to('cpu').numpy())).float(),
    ]

    if is_train:
        x = [
            preprocessed_x[0].to(device),
            preprocessed_x[1].to(device)
        ]
        y = [y[0], y[1]]
    else:
        x = preprocessed_x[2].to(device)
        y = y[2]

    return x, y
