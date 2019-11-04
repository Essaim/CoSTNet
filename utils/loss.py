import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, truth, predict):
        return self.mse_loss(truth, predict) ** 0.5