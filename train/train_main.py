import copy
import torch
import numpy as np
import time

from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from utils.util import save_model
from utils.load_config import get_config


def train_spatio(model,
                 data_loader,
                 normal,
                 loss_func,
                 optimizer,
                 num_epochs,
                 tensorboard_folder: str,
                 model_folder):
    phases = ['train', 'validate']
    since = time.clock()

    writer = SummaryWriter(tensorboard_folder)
    save_dict, best_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 999999
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)

    try:
        for epoch in range(num_epochs):

            running_loss = {phase: 0.0 for phase in phases}
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, groud_truth, prediction = 0, list(), list()
                tqdm_loader = tqdm(data_loader[phase], phase)

                for x, y in tqdm_loader:
                    # x.to(get_config("device"))
                    # y.to(get_config("device"))
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(x)
                        loss = loss_func(y_pred, y)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    groud_truth.append(y.cpu().detach().numpy())
                    prediction.append(y_pred.cpu().detach().numpy())

                    running_loss[phase] += loss * y.size(0)
                    steps += y.size(0)

                    tqdm_loader.set_description(
                        f"{phase:8} epoch: {epoch:3}  loss: {running_loss[phase] / steps:3.6},"
                        f"true loss: {normal.rmse_transform(running_loss[phase] / steps):3.6}")
                    torch.cuda.empty_cache()

                if phase == 'validate' and running_loss[phase] / steps <= best_rmse:
                    best_rmse = running_loss[phase] / steps
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                     epoch=epoch)

            scheduler.step(running_loss['train'])

            writer.add_scalars('Loss',
                               {f'{phase} loss': running_loss[phase] / len(data_loader[phase].dataset) for phase in
                                phases}, epoch)
    finally:
        time_elapsed = time.clock() - since
        print(
            f"cost time: {time_elapsed} seconds          best val loss: {best_rmse}     best epoch: {save_dict['epoch']}")
        save_model(f"{model_folder}/ae_model.pkl", **save_dict)
        model.load_state_dict(save_dict['model_state_dict'])
    return model


def train_temporal(model,
                   data_loader,
                   normal,
                   loss_func,
                   optimizer,
                   num_epochs,
                   tensorboard_folder: str,
                   model_folder: str):
    phases = ['train', 'validate', 'test']
    since = time.clock()
    writer = SummaryWriter(tensorboard_folder)

    loss_func.to(get_config("device"))
    model.to(get_config("device"))

    save_dict, best_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 999999
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)

    try:
        for epoch in range(num_epochs):

            running_loss, running_metrics = {phase: 0.0 for phase in phases}, {phase: dict() for phase in phases}

            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, ground_truth, prediction = 0, list(), list()
                tqdm_loader = tqdm(data_loader[phase], phase)
                for x, y in tqdm_loader:

                    x.to(get_config("device"))
                    y.to(get_config("device"))

                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(x)
                        loss = loss_func(y, y_pred)

                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    ground_truth.append(y.cpu().numpy())
                    prediction.append(y_pred.detach().cpu().numpy())

                    running_loss[phase] += loss * y.size(0)
                    steps += y.size(0)

                    tqdm_loader.set_description(f"{phase:8} epoch: {epoch:8} loss{running_loss[phase] / steps:3.6},"
                                                f"true loss{normal.rmse_transform(running_loss[phase] / steps):3.6}")

                    torch.cuda.empty_cache()

                if phase == 'validate' and running_loss[phase] / len(data_loader[phase].dataset) < best_rmse:
                    best_rmse = running_loss[phase] / len(data_loader[phase].dataset)
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                     epoch=epoch)

            scheduler.step(running_loss['train'])

            writer.add_scalars('Loss',
                               {f'{phase} loss': running_loss[phase] / len(data_loader[phase].dataset) for phase in
                                phases}, epoch)
    finally:
        time_elapsed = time.clock() - since
        print(
            f"cost time: {time_elapsed} seconds          best val loss: {best_rmse}     best epoch: {save_dict['epoch']}")
        save_model(f"{model_folder}/best_model.pkl", **save_dict)
        model.load_state_dict(save_dict['model_state_dict'])
    return model
