#Necessary built-in modules
import random
import os
#Pytorch modules
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#PIL modules
from PIL import Image
#Pre-written modules
from UMD PRG Hoop Recognition Model/Image Transformation Modules/generate_hoop_image.py

#Generate hoop
hoop = transform_image(Image.open("assets/hoops/hoop1.png"))

#Get random background image
background = Image.open("assets/unlabeled2017/" + random.choice(os.listdir("assets/unlabeled2017")))

#Get background modified
result = get_image(background, hoop)
result[0].show()
result[1].show()

print(os.listdir("assets/unlabeled2017"))