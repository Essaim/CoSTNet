import torch
from torch import nn
import numpy


class Model_Predict_LSTM(nn.Module):
    def __init__(self, encoder, decoder, input_size, output_size ):

        super(Model_Predict_LSTM, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        x = self.decoder(x)
        output, (h_n, c_n) = self.lstm(x)
        return self.decoder(h_n[-1, :, :])


