#Necessary built-in modules
import random
import os
import numpy as np
import sys
sys.path.append('/training') 
#Pytorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
#PIL modules
from PIL import Image
#Pre-written modules
from training import hoop_dataset

#Create dataset
dataset = hoop_dataset.HoopDataset("/unlabeled2017", "/hoops")