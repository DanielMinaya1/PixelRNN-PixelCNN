import torch
from torch import nn

from utils.config import device
from utils.maskedConv import MaskedConv1d

class RowLSTMLayer(nn.Module):
	def __init__(self, in_channels, hidden_dim, image_size, device=device):
		super(RowLSTMLayer, self).__init__()

		self.hidden_dim = hidden_dim
		self.image_size = image_size
		self.in_channels = in_channels
		self.num_units = self.hidden_dim * self.image_size
		self.output_size = self.num_units
		self.state_size = self.num_units * 2

		self.conv_i_s = MaskedConv1d('B', hidden_dim, 4*self.hidden_dim, kernel_size=3, stride=1, padding=1)
		self.conv_s_s = nn.Conv1d(in_channels, 4*self.hidden_dim, kernel_size=3, padding=1, stride=1)

		self.device = device
		self.to(self.device)

	def forward(self, inputs, states):
		c_prev, h_prev = states

		h_prev = h_prev.view(-1, self.hidden_dim,  self.image_size)
		inputs = inputs.view(-1, self.in_channels, self.image_size)

		s_s = self.conv_s_s(h_prev)
		i_s = self.conv_i_s(inputs)

		s_s = s_s.view(-1, 4 * self.num_units) 
		i_s = i_s.view(-1, 4 * self.num_units)

		lstm = s_s + i_s
		lstm = torch.sigmoid(lstm)

		i, g, f, o = torch.split(lstm, (4 * self.num_units)//4, dim=1)
		c = f * c_prev + i * g
		h = o * torch.tanh(c)

		new_state = (c, h)
		return h, new_state

class RowLSTM(nn.Module):
	def __init__(self, in_channels, hidden_dim, input_size, device=device):
		super(RowLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.init_state = (torch.zeros(1, input_size * hidden_dim).to(device), torch.zeros(1, input_size * hidden_dim).to(device))
		self.lstm_layer = RowLSTMLayer(in_channels, hidden_dim, input_size)

		self.device = device
		self.to(self.device)

	def forward(self, inputs, initial_state=None):
		n_batch, channel, n_seq, width = inputs.size()

		if initial_state is None:
			hidden_init, cell_init = self.init_state
		else:
			hidden_init, cell_init = initial_state

		states = (hidden_init.repeat(n_batch,1), cell_init.repeat(n_batch, 1))

		steps = []
		for seq in range(n_seq):
			h, states = self.lstm_layer(inputs[:, :, seq, :], states)
			steps.append(h.unsqueeze(1))

		output = torch.cat(steps, dim=1)
		output = output.view(-1, n_seq, width, self.hidden_dim)
		output = output.permute(0,3,1,2)
		return output