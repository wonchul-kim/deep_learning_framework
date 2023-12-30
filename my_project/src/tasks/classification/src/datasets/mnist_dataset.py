import torch 
import torchvision 

train_dataset = torchvision.datasets.MNIST('/HDD/datasets/tmp', train=True, download=True)
test_dataset = torchvision.datasets.MNIST('/HDD/datasets/tmp', train=False, download=True)

train_data = train_dataset.train_data
train_labels = train_dataset.train_labels
