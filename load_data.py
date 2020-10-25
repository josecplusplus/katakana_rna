from PIL import Image
import numpy as np
import os

train_path = "./bitmaps/train/"
test_path = "./bitmaps/test/"


test_images = []
train_images = []



def init_train_images():
    return load_from_disk(train_path, train_images)

def init_test_images():
    return load_from_disk(test_path, test_images)

def load_from_disk(path, images_array):
    for root, dirs, files in os.walk(train_path):
        for filename in files:
            images_array.append(load_image_as_array(path + filename))
    return images_array
    
def load_image_as_array(filename):
    image = Image.open(filename)
    return np.asarray(image)
    
    
    
def get_pattern_length():
    return (64)
