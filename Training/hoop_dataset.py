#Built-in modules
import numpy as np
import random
import os
import sys
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#Image modules
from PIL import Image
#Prewritten modules
import generate_training_data
#import image_transformations 
# from .generate_hoop_image import transform_image
# from .generate_training_data import add_hoop_to_background

class HoopDataset(Dataset): 

    def __init__(self, hoop_dir, background_dir, transform=None):
        """
        Args:
            hoop_dir (string): Path to directory with all hoops
            background_dir (string): Path to directory with all backgrounds
        """
        self.background_dir = background_dir
        self.hoop_dir = hoop_dir
        self.hoops = os.listdir(hoop_dir)
        self.backgrounds = os.listdir(background_dir)
        self.num_backgrounds = len(self.backgrounds)
        self.data = []
        self.transform = transform
        self.__annotations__ = []
        
    def __len__(self):
        return(len(self.data))

    def __getitem__(self, idx):
        #Get base hoop
        hoop = Image.open(self.hoop_dir + "/" + self.hoops[idx % len(self.hoops)])    
        #Get base background
        background = Image.open(self.background_dir + "/" + self.backgrounds[idx % self.num_backgrounds])
        data = generate_training_data.add_hoop_to_background(background, hoop)
        #transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

        while(idx >= len(self.__annotations__)):
            self.__annotations__.append(None)
            self.data.append(None)
        self.__annotations__[idx] = data[0]
        self.data[idx] = data[1]
        if self.transform is not None:
            data = [self.transform(item) for item in data]

        tf=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512,640)),
            transforms.ToTensor()
        ])
        data[0] = tf(data[0])
        data[1] = tf(data[1])
        return(data[0], data[1])