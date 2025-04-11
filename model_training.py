import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import os

import matplotlib.pyplot as plt

from CNN import CNN

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    model = model.to(device)
    for epoch in range(epochs):

        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

# def train_with_krylov(model, device, train_loader, optimizer, criterion, epochs):
#     model.train()
#     model = model.to(device)
#     for epoch in range(epochs):
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.model.zero_grad()
#             outputs = model(images)
#
#             loss = criterion(outputs, labels)
#
#             loss.backward()
#             optimizer.step(loss, model.parameters)

def test(model, device, test_loader):
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    #Individual accuracy for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the model on the 10000 test images: {100 * correct // total} %')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    PATH = './cifar_model.pth'

    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train(model, device, train_loader, optimizer, criterion, epochs=10)

        torch.save(model.state_dict(), PATH)

    test(model, device, test_loader)




    
