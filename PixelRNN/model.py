import torch
from torch import nn

from PixelRNN.RowLSTM import RowLSTMLayer, RowLSTM
from PixelRNN.DiagLSTM import DiagonalLSTM
from utils.maskedConv import MaskedConv2d
from utils.config import device

class PixelRNN_model(nn.Module):
    def __init__(self, num_layers, num_filters, input_size, type_rnn='row', device=device):
        super(PixelRNN_model, self).__init__()

        self.conv1 = MaskedConv2d('A', 1, num_filters, kernel_size=7, stride=1, padding=3)
        if type_rnn=='row':
          self.lstm_list = nn.ModuleList([RowLSTM(num_filters, num_filters, input_size) for _ in range(num_layers)])
        elif type_rnn=='diagonal':
          self.lstm_list = nn.ModuleList([DiagonalLSTM(num_filters, num_filters) for _ in range(num_layers)])
        self.conv2 = MaskedConv2d('B', num_filters, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = MaskedConv2d('B', 32, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        x = self.conv1(inputs)
        for lstm in self.lstm_list:
            x = lstm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x