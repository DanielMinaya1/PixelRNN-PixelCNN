import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np

from utils.config import batch_size_train, batch_size_test

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

def get_cifar10_dataset(train=True):
    return datasets.CIFAR10(
        root='data',
        train=train,
        transform=transform,
        download=True if train else False
    )

train_dataset = get_cifar10_dataset(train=True)
test_dataset = get_cifar10_dataset(train=False)

train_dataset = np.transpose(train_dataset.data, [0, 3, 1, 2]) / 255
test_dataset = np.transpose(test_dataset.data, [0, 3, 1, 2]) / 255

train_loader_cifar10 = data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader_cifar10 = data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)