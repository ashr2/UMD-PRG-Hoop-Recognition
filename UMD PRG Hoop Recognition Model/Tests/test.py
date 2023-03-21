import numpy as np
import torch
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
from add_image_to_background import get_image
from torch.utils.data import Dataset, DataLoader
#import hoop_dataset
from PIL import Image

#Generate hoop
hoop = transform_image(Image.open("assets/hoops/hoop1.png"))

#Get random background image
background = Image.open("assets/unlabeled2017/" + random.choice(os.listdir("assets/unlabeled2017")))

#Get background modified
result = get_image(background, hoop)
result[0].show()
result[1].show()

print(os.listdir("assets/unlabeled2017"))