import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.config import device, criterion, lr

def train(model, train_loader, test_loader, epochs, optimizer):
	train_losses = []
	init_test_loss = evaluate(model, test_loader)
	test_losses = [init_test_loss]

	for epoch in range(epochs):
		model.train()
		for batch in train_loader:
			batch = batch.float().to(device)
			optimizer.zero_grad()
			output = model(batch)
			loss = criterion(output, batch)
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())
		test_losses.append(evaluate(model, test_loader))
		print(f'Test loss: {test_losses[-1]}')
		print(f'{epoch + 1}/{epochs} epochs')

	return np.array(train_losses), np.array(test_losses)

def evaluate(model, iterator):
	model.eval()
	tot_loss = []
	with torch.no_grad():
		for batch in iterator:
			batch = batch.float().to(device)
			output = model(batch)
			tot_loss.append(criterion(output, batch).item())
		return np.mean(np.array(tot_loss))

def sampling(model, num_samples, image_shape=(28,28)):
	model.eval()
	H, W = image_shape
	samples = torch.zeros(size=(num_samples, 1, H, W)).to(device)
	with torch.no_grad():
		for i in range(H):
			for j in range(W):
				out = model(samples)
				samples[:, :, i, j] = torch.bernoulli(out[:, :, i, j], out=samples[:, :, i, j])

	return samples

def plot_samples(model, n_samples):
	sample = sampling(model, num_samples=n_samples)
	sample = sample.cpu()

	if n_samples<10:
		fig, axes = plt.subplots(1, n_samples)
		for i, ax in enumerate(axes.flatten()):
			ax.imshow(sample[i][0], cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])

	else:
		fig, axes = plt.subplots(n_samples//10, 10, figsize=(16,16))
		for i, ax in enumerate(axes.flatten()):
			ax.imshow(sample[i][0], cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])

def experiment(model, epochs, train_loader, test_loader):
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	train_losses, test_losses = train(model, train_loader, test_loader, epochs, optimizer)
	return train_losses, test_losses