import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
criterion = lambda x,y: F.binary_cross_entropy(x,y)

batch_size_train = 16
batch_size_test = 128