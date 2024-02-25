# this file is to resize images to the same size, without maintaining their aspect ratio
from PIL import Image
import os

thisPath = os.getcwd()
training_filenames = os.listdir(thisPath + "\\train")

for name in training_filenames:
    img = Image.open("train\\" + name)
    img = img.resize((64, 64))
    img.save("train_resized\\" + name, "JPEG")

# you can use transforms in torchvision.transforms.Resize to achieve something similar without saving new images
