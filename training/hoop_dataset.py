#Built-in modules
import cv2
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
import PIL
from PIL import Image
#Transformation modules
from generate_hoop_image import transform_image
from generate_training_data import add_hoop_to_background

class HoopDataset(Dataset): 

    def __init__(self, hoop_dir, background_dir, transform=None):
        """
        Args:
            hoop_dir (string): Path to directory with all hoops
            background_dir (string): Path to directory with all backgrounds
        """
        self.background_dir = background_dir
        self.hoop_dir = hoop_dir
        self.hoops = sorted(os.listdir(hoop_dir))
        self.backgrounds = sorted(os.listdir(background_dir))
        self.backgrounds.remove(".DS_Store")
        self.num_backgrounds = len(self.backgrounds)
        self.transform = transform
        self.cache = {} # TODO cache to disk instead of RAM

    def __len__(self):
        return self.num_backgrounds * len(self.hoops)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        #Get base hoop
        hoop_idx = idx % len(self.hoops)
        back_idx = int(idx / len(self.hoops)) % self.num_backgrounds
        hoop = Image.open(self.hoop_dir + "/" + self.hoops[hoop_idx])

        #Get base background
        background = Image.open(self.background_dir + "/" + self.backgrounds[back_idx])

        #Create image to be used for training
        hoop_back, hoop_black = add_hoop_to_background(background, hoop)
        tf=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,640)),
        ])
        hoop_black = hoop_black.convert("RGB")
        hoop_back  = tf(hoop_back)
        hoop_black = tf(hoop_black)
        hoop_black = (hoop_black > 0).type(torch.float)
        data = (hoop_back, hoop_black)
        if self.transform is not None:
           data = [self.transform(item) for item in data]

        #Add image to cache
        self.cache[idx] = (idx, *data)

        return idx, *data