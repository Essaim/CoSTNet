import os
import torch
from utils.load_config import get_config


def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


def tensor2numpy(ten):
    return ten.cpu().detach().numpy()

def numpy2tensor(num):
    return torch.from_numpy(num).float().to(get_config("device"))