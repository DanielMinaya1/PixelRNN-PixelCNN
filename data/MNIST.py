import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from utils.config import batch_size_train, batch_size_test

def get_mnist_dataset(train=True):
    return datasets.MNIST(
        root='data',
        train=train,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
        download=True if train else False
    )

train_dataset = get_mnist_dataset(train=True)
test_dataset = get_mnist_dataset(train=False)

train_dataset = train_dataset.data.unsqueeze(1) / 255
test_dataset = test_dataset.data.unsqueeze(1) / 255

train_loader_mnist = data.DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader_mnist = data.DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)