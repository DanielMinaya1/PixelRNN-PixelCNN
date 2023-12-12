import torch
from torch import nn

from utils.config import device
from utils.maskedConv import MaskedConv2d 
from utils.maskedConv import ResidualMaskedConv2d

class PixelCNN_model(nn.Module):
	def __init__(self, num_channels, num_filters, num_layers, device=device):
		super().__init__()
		self.conv1 = MaskedConv2d('A', num_channels, 2*num_filters, kernel_size=7, stride=1, padding=3)
		self.relu = nn.ReLU()
		self.conv_list = nn.ModuleList([nn.Sequential(
			ResidualMaskedConv2d(2*num_filters), 
			nn.ReLU(), 
			nn.BatchNorm2d(2*num_filters)
			) for _ in range(num_layers)])
		self.conv2 = MaskedConv2d('B', 2*num_filters, num_channels, kernel_size=1, padding=0)
		self.sigmoid = nn.Sigmoid()

		self.device = device
		self.to(self.device)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		for conv_layer in self.conv_list:
			x = conv_layer(x)
		x = self.conv2(x)
		x = self.sigmoid(x)
		return x