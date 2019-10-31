import numpy as np

import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import random

from utils.load_config import get_config


def get_pretrain_dataloader(datapath: str,
                            batch_size: int,
                            channel: int,
                            pre_train_len: int):
    np.random.seed(10)
    data = h5py.File(datapath)['data'][:, channel]
    np.random.shuffle(data)
    split = len(data) * pre_train_len
    return {'train': DataLoader(dataset=TensorDataset(data[:split], data[:split]), shuffle=True, batch_size=batch_size),
            'validate': DataLoader(dataset=TensorDataset(data[split:], data[split:]), shuffle=True,
                                   batch_size=batch_size)}


class predict_dataset(Dataset):
    def __init__(self, x, y, key):
        self.x = x
        self.y = y
        self.key = key

    def __getitem__(self, item: int):
        if self.key == 'train':
            return self.x[:get_config("train_len")], self.y[:get_config("train_len")]
        elif self.key == 'validate':
            return self.x[get_config("train_len"): get_config("train_len") + get_config("validate_len")]
        elif self.key == 'test':
            return self.x[-get_config("test_len"):]

    def __len__(self):
        return get_config(f"{self.key}_len")


def get_predict_dataloader(datapath: str,
                           batch_size: int,
                           channel: int,
                           depend_list: list):
    data = h5py.File(datapath)['data'][:, channel]

    X_, Y_ = list(), list()
    for i in range(depend_list[0], data.shape[0]):
        X_.append([data[i - j] for j in depend_list])
        Y_.append(data[i])

    X_ = torch.from_numpy(np.asarray(X_)).float().to(get_config('device'))
    Y_ = torch.from_numpy(np.asarray(Y_)).float().to(get_config('device'))

    dls = dict()
    for key in ['train', 'validate', 'test']:
        dataset = predict_dataset(X_, Y_, key)
        dls[key] = DataLoader(dataset=dataset,
                              shuffle=True,
                              batch_size=batch_size)

    return dls