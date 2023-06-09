from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
import torchvision.transforms as T

def transform_image(orig_img):
    transformed_img = orig_img
    #Rotate image
    rotater = T.RandomRotation(degrees=(0, 180))
    rotated_img = [rotater(transformed_img)]
    transformed_img = rotated_img[0]

    #Flip vertically
    vflipper = T.RandomVerticalFlip(p=0.5)
    transformed_imgs = [vflipper(transformed_img)]
    transformed_img = transformed_imgs[0]

    #Flip horizontally
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(transformed_img)]
    transformed_img = transformed_imgs[0]

    #Perspective image
    perspective_transformer = T.RandomPerspective(distortion_scale=0.8, p=1.0) #Test distortion scale, play around with it
    perspective_imgs = [perspective_transformer(transformed_img)]
    transformed_img = perspective_imgs[0]
    
    return(transformed_img)
