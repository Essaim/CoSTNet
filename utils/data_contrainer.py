import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn


from utils.normalization import MinMaxNormal
from utils.load_config import get_config
from utils.util import tensor2numpy, numpy2tensor


def get_spatio_dataloader(datapath: str,
                          normal: MinMaxNormal,
                          batch_size: int,
                          channel: int,
                          pre_train_len: int):
    np.random.seed(10)
    data = h5py.File(datapath)['data'][:, channel]
    data = normal.transform(data)
    np.random.shuffle(data)
    vali_len = int(len(data) * pre_train_len)
    data = torch.from_numpy(data).float().to(get_config("device")).unsqueeze(-3)
    return {'train': DataLoader(dataset=TensorDataset(data[:-vali_len], data[:-vali_len]), shuffle=True,
                                batch_size=batch_size),
            'validate': DataLoader(dataset=TensorDataset(data[-vali_len:], data[-vali_len:]), shuffle=True,
                                   batch_size=batch_size)}


class temporal_dataset(Dataset):
    def __init__(self, x, y, key, train_len, validate_len, test_len):
        self.x = x
        self.y = y
        self.key = key
        self.len = {"train_len": train_len - validate_len,
                    "validate_len": validate_len, "test_len": test_len}

    def __getitem__(self, item: int):
        if self.key == 'train':
            return self.x[item], self.y[item]
        elif self.key == 'validate':
            return self.x[self.len["train_len"] + item], self.y[self.len["train_len"] + item]
        elif self.key == 'test':
            return self.x[-self.len["test_len"] + item], self.y[-self.len["test_len"] + item]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.len[f"{self.key}_len"]


def get_temporal_dataloader(data_input: list,
                            normal: list,
                            batch_size: int,
                            encoder: nn.Module,
                            depend_list: list):
    data_ground, data_encode, channel_num = list(), list(), len(data_input)
    for i in range(channel_num):
        data_ = np.expand_dims(data_input[i], axis=-3)
        data_ = normal[i].transform(data_)
        data_encode_ = tensor2numpy(encoder[i](numpy2tensor(data_)))

        data_ground.append(data_)
        data_encode.append(data_encode_)


    data_encode = np.concatenate([each for each in data_encode], axis=1)
    data_ground = np.concatenate([each for each in data_ground], axis=1)
    X_, Y_ = list(), list()
    for i in range(depend_list[0], data_encode_.shape[0]):
        X_.append([data_encode[i - j] for j in depend_list])
        Y_.append(data_ground[i])

    X_ = torch.from_numpy(np.asarray(X_)).float().to(get_config('device'))
    Y_ = torch.from_numpy(np.asarray(Y_)).float().to(get_config('device'))
    dls = dict()
    for key in ['train', 'validate', 'test']:
        dataset = temporal_dataset(X_, Y_, key, get_config("temporal_train_len"), get_config("temporal_validate_len"),
                                   get_config("temporal_test_len"))
        dls[key] = DataLoader(dataset=dataset,
                              shuffle=True,
                              batch_size=batch_size)

    return dls
