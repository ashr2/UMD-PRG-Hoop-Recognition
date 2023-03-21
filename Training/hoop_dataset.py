import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as T
from torchvision import datasets, transforms, models
from add_image_to_background import get_image
from torch.utils.data import Dataset, DataLoader
from tranforms import transform_image

class HoopDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, hoop_dir, background_dir, size, transform=None):
        """
        Args:
            hoop_dir (string): Path to directory with all hoops
            background_dir (string): Path to directory with all backgrounds
            size (int): number of images in training set
        """
        self.hoops = os.listdir(hoop_dir)
        self.backgrounds = os.listdir(background_dir)
        self.data = []
        self.__annotations__ = []
        #Generate size number of images
        for i in range(size):
            #Get base hoop
            hoop = ".DS_Store"
            while(hoop == ".DS_Store"):
                hoop = random.choice(self.hoops)
            #Get base background
            background = ".DS_Store"
            while(background == ".DS_Store"):
                background = random.choice(self.backgrounds)
            data = get_image(Image.open(background_dir + "/" + background), Image.open(hoop_dir + "/" + hoop))
            self.__annotations__.append(data[0])
            self.data.append(data[1])
            
    def __len__(self):
        return(self.size)

    def __getitem__(self, idx):
        return(self.__annotations__[idx], self.data[idx])