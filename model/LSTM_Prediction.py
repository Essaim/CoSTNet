import torch
from torch import nn
import numpy


class Model_Temporal_LSTM(nn.Module):
    def __init__(self, encoder, decoder, lstm_width, lstm_height, channel_num):
        super(Model_Temporal_LSTM, self).__init__()

        self.decoder = decoder
        for p in self.parameters():
            p.requires_grad = False

        self.width = lstm_width
        self.height = lstm_height
        self.channel_num = channel_num
        self.lstm = nn.LSTM(input_size=lstm_width * lstm_height * channel_num,
                            hidden_size=lstm_width * lstm_height* channel_num,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        output, (h_n, c_n) = self.lstm(x)
        lstm_out = h_n[-1, :, :].view(h_n[-1, :, :].shape[0], self.channel_num, self.height, self.width)
        decoder_out = [self.decoder[i](lstm_out[:,i,:,:].unsqueeze(-3)) for i in range(self.channel_num)]
        return torch.cat(decoder_out,-3)
