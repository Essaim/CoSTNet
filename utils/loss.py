import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, truth, predict):
        return self.mse_loss(truth, predict) ** 0.5

def loss_calculate(y, y_pred, running_loss, phase, loss_func, channel_num):
    for i in range(channel_num):
        running_loss[phase][i] += loss_func(y[:,i], y_pred[:,i])* y.size(0)
    running_loss[phase][channel_num] += loss_func(y, y_pred) * y.size(0)
    return running_loss