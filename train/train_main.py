import copy
import torch
import numpy as np
import time

from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from utils.util import save_model
from utils.load_config import get_config
from utils.loss import loss_calculate


def train_model(model,
                data_loader,
                phases,
                normal,
                loss_func,
                optimizer,
                num_epochs,
                tensorboard_folder: str,
                model_folder_name):
    since = time.clock()

    writer = SummaryWriter(tensorboard_folder)
    save_dict, best_rmse, best_test_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()),
                                            'epoch': 0}, 999999, 999999
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
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()), epoch=epoch)
                if phase == 'test' and save_dict['epoch'] == epoch:
                    best_test_rmse = running_loss[phase] / steps

            scheduler.step(running_loss['train'])

            writer.add_scalars(f'Loss',
                               {f'{phase} loss': running_loss[phase] / len(data_loader[phase].dataset) for phase in
                                phases}, epoch)
    finally:
        time_elapsed = time.clock() - since
        print(f"cost time: {time_elapsed:.2} seconds   best val loss: {normal.rmse_transform(best_rmse)}   "
              f"best test loss:{normal.rmse_transform(best_test_rmse)}  best epoch: {save_dict['epoch']}")
        save_model(f"{model_folder_name}", **save_dict)
        # model.load_state_dict(torch.load(f"{model_folder_name}")['model_state_dict'])
        model.load_state_dict(save_dict['model_state_dict'])
    return model


def train_temporal_model(model: nn.Module,
                         data_loader,
                         phases: list,
                         normal: list,
                         channel_list: list,
                         channel: int,
                         loss_func: nn.Module,
                         optimizer,
                         num_epochs: int,
                         tensorboard_folder: str,
                         model_folder_name):
    since = time.clock()
    channel_num = len(channel_list)

    writer = SummaryWriter(tensorboard_folder)
    save_dict, best_rmse, best_test_rmse = [{'model_state_dict': copy.deepcopy(model.state_dict()),
                                             'epoch': 0} for i in range(channel_num)], \
                                           [999999 for i in range(channel_num)], \
                                           [999999 for i in range(channel_num)]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, threshold=1e-3, min_lr=1e-6)

    try:
        for epoch in range(num_epochs):

            running_loss = {phase: [0.0 for i in range(channel_num + 1)] for phase in phases}
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

                    # running_loss[phase] += loss * y.size(0)
                    running_loss = loss_calculate(y, y_pred, running_loss, phase, loss_func, channel_num)
                    steps += y.size(0)

                    tqdm_loader.set_description(
                        f"{phase:8} epoch: {epoch:3} loss: {running_loss[phase][channel_num] / steps:3.6},"
                        f"true loss: {normal[channel].rmse_transform(running_loss[phase][channel_num] / steps):3.6}")
                    torch.cuda.empty_cache()

                for i in range(channel_num):
                    if phase == 'validate' and running_loss[phase][i] / steps <= best_rmse[i]:
                        best_rmse[i] = running_loss['validate'][i] / steps
                        save_dict[i].update(model_state_dict=copy.deepcopy(model.state_dict()), epoch=epoch)
                    if phase == 'test' and save_dict[i]['epoch'] == epoch:
                        best_test_rmse[i] = running_loss['test'][i] / steps

            scheduler.step(running_loss['train'][channel_num])

            # steps = len(data_loader['train'].dataset)    cuowu
            for i in range(channel_num):
                writer.add_scalars(f'Loss channel{channel_list[i]}',
                                   {f'{phase} true loss': normal[i].rmse_transform(
                                       running_loss[phase][i] / len(data_loader[phase].dataset))
                                    for phase in phases}, epoch)
            writer.add_scalars(f'Loss channel total',
                               {f'{phase} loss': normal[channel].rmse_transform(
                                   running_loss[phase][channel_num] / len(data_loader[phase].dataset))
                                for phase in phases}, epoch)
    finally:
        time_elapsed = time.clock() - since
        print(f"cost time: {time_elapsed:.2} seconds")
        for i in range(channel_num):
            save_model(f"{model_folder_name}_{channel_list[i]}.pkl", **save_dict[i])
        # model.load_state_dict(torch.load(f"{model_folder_name}")['model_state_dict'])
        # model.load_state_dict(save_dict['model_state_dict'])
    # return model
