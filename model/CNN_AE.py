import numpy as np
from torch import nn
import torch
import h5py



class Model_Spatio_CNN(nn.Module):
    def __init__(self, filter_num):

        super(Model_Spatio_CNN, self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(1, filter_num[0], (3, 3), stride=1, padding=1),
                                     nn.Conv2d(filter_num[0], filter_num[1], (3, 3), stride=1, padding=1),
                                     nn.Conv2d(filter_num[1], 1, (3, 3), stride=1, padding=1))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(1, filter_num[1], (3, 3), stride=1, padding=1),
                                     nn.ConvTranspose2d(filter_num[1], filter_num[0], (3, 3), stride=1, padding=1),
                                     nn.ConvTranspose2d(filter_num[0], 1, (3, 3), stride=1, padding=1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x