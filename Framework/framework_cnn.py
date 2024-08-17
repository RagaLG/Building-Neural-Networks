# -*- coding: utf-8 -*-
"""framework_CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gVi-GkptiNEiYoD-RbjwmZJN2ONXQWBJ
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def main():

    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())


    classes = dataset.classes
    class_count = {}
    for _, index in dataset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1

    torch.manual_seed(43)
    val_size = 5000
    train_size = 45000
    print()

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_ds)}, Validation set size: {len(val_ds)}, Test set size: {len(test_dataset)}")

    batch_size=128  # set batch_size
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4)

    device = torch.device('cpu') # default device
    print(f"Device: {device}\n")
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)

    model = to_device(CIFAR_NN(), device)
    num_epochs = 20
    lr = 0.001

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Count of parameters: {count_parameters(CIFAR_NN())}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print()

    history = fit(num_epochs, lr, model, train_dl, val_dl, torch.optim.Adam)
    print()

    test_dl = DeviceDataLoader(test_loader, device)
    test_res = evaluate(model, test_dl)
    print(f"Test Loss: {test_res['val_acc']:.4f}, Test set accuracy: {test_res['val_acc']:.4f}")

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch {epoch} - validation_loss: {result['val_loss']}, validation_accuracy: {result['val_acc']}")

class CIFAR_NN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(9)
        self.Conv2 = nn.Conv2d(9, 9, kernel_size=3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(9)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv3 = nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(18)
        self.Conv4 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        self.BatchNorm4 = nn.BatchNorm2d(18)
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv5 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1)
        self.BatchNorm5 = nn.BatchNorm2d(36)
        self.Conv6 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        self.BatchNorm6 = nn.BatchNorm2d(36)
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(576, 100)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(100, 10)


    def forward(self, xb):
        xb = F.relu(self.Conv1(xb))
        xb = self.BatchNorm1(xb)
        xb = F.relu(self.Conv2(xb))
        xb = self.BatchNorm2(xb)
        xb = self.MaxPool_1(xb)

        xb = F.relu(self.Conv3(xb))
        xb = self.BatchNorm3(xb)
        xb = F.relu(self.Conv4(xb))
        xb = self.BatchNorm4(xb)
        xb = self.MaxPool_2(xb)

        xb = F.relu(self.Conv5(xb))
        xb = self.BatchNorm5(xb)
        xb = F.relu(self.Conv6(xb))
        xb = self.BatchNorm6(xb)
        xb = self.MaxPool_3(xb)

        xb = self.flatten(xb)
        xb = F.relu(self.linear(xb))
        xb = self.dropout(xb)
        xb = self.output(xb)
        return xb

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)
    print(f"Optimizer: {optimizer}\n")
    for epoch in range(epochs):

        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


if __name__ == '__main__':
    main()

