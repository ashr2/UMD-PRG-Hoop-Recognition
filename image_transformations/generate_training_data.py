import PIL
from PIL import Image
import random
import os
import cv2

def add_hoop_to_background(background_image, hoop_image):
    BLACK_IMAGE_PATH = "assets/black.jpg"
    black_image = Image.open(BLACK_IMAGE_PATH)

    size = random.randint(100,min(background_image.size[1], background_image.size[0])//2)
    maxsize = (size, size)
    hoop_image.thumbnail(maxsize, PIL.Image.ANTIALIAS)

    #Paste hoop onto background
    x_center = random.randint(hoop_image.size[0], background_image.size[0] - hoop_image.size[0])
    y_center = random.randint(hoop_image.size[1], background_image.size[1] - hoop_image.size[1])
    #im.paste(mouse, (40,40), mouse)
    hoop_mask = hoop_image.convert("L")
    background_image.paste(hoop_image, (x_center, y_center), hoop_mask)

    #Paste hoop onto black background
    black_image.paste(hoop_image, (x_center, y_center))
    black_image = black_image.crop((0, 0, background_image.size[0], background_image.size[1]))
    return(black_image, background_image)