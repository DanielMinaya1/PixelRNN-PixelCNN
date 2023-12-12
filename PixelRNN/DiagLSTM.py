import torch
from torch import nn

from utils.config import device
from utils.maskedConv import MaskedConv2d

def skew(tensor):
    B, C, H, W = tensor.shape
    output = tensor.new_zeros((B, C, H, H+W-1))
    for row in range(H):
        columns = (row, row + W)
        output[:, :, row, columns[0]:columns[1]] = tensor[:, :, row]
    return output

def unskew(tensor):
    B, C, H, skew_W = tensor.shape
    W = skew_W - H + 1
    output = tensor.new_zeros((B, C, H, W))
    for row in range(H):
        columns = (row, row + W)
        output[:, :, row] = tensor[:, :, row, columns[0]:columns[1]]
    return output

class DiagonalLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, device=device):
        super(DiagonalLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.conv_is = MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=5 * hidden_dim, kernel_size=1, padding=0)
        self.conv_ss = nn.Conv1d(hidden_dim, 5 * hidden_dim, [2], padding=1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.device = device
        self.to(self.device)

    def forward(self, inputs):
        skewed_input = skew(inputs)
        B, C, H, skew_W = skewed_input.shape

        i_s = self.conv_is(skewed_input)

        hStates = []
        cStates = []

        h_prev = skewed_input.new_zeros([B, self.hidden_dim, H])
        c_prev = skewed_input.new_zeros([B, self.hidden_dim, H])

        for i in range(skew_W):
            input_column = skewed_input[..., i]
            s_s = self.conv_ss(h_prev)[..., :-1]
            gates = i_s[..., i] + s_s

            o, f_left, f_up, i, g = torch.chunk(gates, 5, dim=1)
            o, f_left, f_up, i, g = self.sigmoid(o), self.sigmoid(f_left), self.sigmoid(f_up), self.sigmoid(i), self.tanh(g)

            c_prev_shifted = torch.cat([input_column.new_zeros([input_column.shape[0], self.hidden_dim,  1]), c_prev], 2)[..., :-1]
            c = (f_left * c_prev + f_up * c_prev_shifted) + i * g
            h = o * self.tanh(c)

            hStates.append(h)
            cStates.append(c)

            h_prev = h
            c_prev = c

        total_hStates = unskew(torch.stack(hStates, dim=3))
        total_cStates = unskew(torch.stack(cStates, dim=3))

        return total_hStates