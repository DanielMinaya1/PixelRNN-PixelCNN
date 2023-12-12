import torch
from torch import nn

from utils.config import device

class MaskedConv1d(nn.Conv1d):
	def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=0, device=device):
		super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)

		kernel_Width = kernel_size
		self.mask = torch.ones(out_channels, in_channels, kernel_Width).to(device)
		self.mask[:, :, kernel_Width // 2 + (mask_type == 'B'):] = 0

		self.device = device
		self.to(self.device)

	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv1d, self).forward(x)

class MaskedConv2d(nn.Conv2d):
	def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=1, device=device):
		super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)

		kernel_Height, kernel_Width = kernel_size, kernel_size
		self.mask = torch.ones(out_channels, in_channels, kernel_Height, kernel_Width).to(device)
		self.mask[:, :, kernel_Height // 2, kernel_Width // 2 + (mask_type == 'B'):] = 0
		self.mask[:, :, kernel_Height // 2 + 1:] = 0

		self.device = device
		self.to(self.device)

	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv2d, self).forward(x)

class ResidualMaskedConv2d(nn.Module):
	def __init__(self, input_dim, device=device):
		super(ResidualMaskedConv2d, self).__init__()

		self.net = nn.Sequential(
			MaskedConv2d('B', input_dim, input_dim // 2, kernel_size=1, padding=0),
			nn.ReLU(),
			MaskedConv2d('B', input_dim // 2, input_dim // 2, kernel_size=3, padding=1),
			nn.ReLU(),
			MaskedConv2d('B', input_dim // 2, input_dim, kernel_size=1, padding=0)
		)

		self.device = device
		self.to(self.device)

	def forward(self, x):
		return self.net(x) + x