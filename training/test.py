#Necessary built-in modules
import random
import os
import numpy as np
import sys
import os
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#PIL modules
from PIL import Image
import hoop_dataset
import encoder
# import generate_training_data
# import training
#Pre-written modules    
#Create dataset
# Define custom collate function
def custom_collate(batch):
    inputs = []
    targets = []
    for item in batch:
        input_item = item[0]
        target_item = item[1]
        if isinstance(input_item, torch.Tensor):
            input_item = transforms.ToPILImage()(input_item)
        if isinstance(target_item, torch.Tensor):
            target_item = transforms.ToPILImage()(target_item)
        inputs.append(input_item)
        targets.append(transforms.ToTensor()(target_item))
    return inputs, targets


# Create dataset
dataset = hoop_dataset.HoopDataset("/Users/ashwathrajesh/UMD-PRG-Hoop-Recognition/tests/hoops", "/Users/ashwathrajesh/UMD-PRG-Hoop-Recognition/assets/unlabeled2017")
# Pre-load dataset
for i in range(0,40):
    dataset[i]

# Set batch size
batch_size = 1

# Create data loader with custom collate function
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Set up autoencoder and optimizer
autoencoder = encoder.AutoEncoder(in_channels=1, out_channels=1)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        print(inputs)
        optimizer.zero_grad()
        inputs = torch.stack([transforms.ToTensor()(x) for x in inputs])
        outputs = autoencoder(inputs)
        loss = F.binary_cross_entropy(outputs, inputs)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
