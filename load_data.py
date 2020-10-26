from PIL import Image
import numpy as np
import os

train_path = "./bitmaps/train_128x128/"
test_path = "./bitmaps/test_128x128/"


test_images = []
train_images = []
test_set = []


def init_train_images():
    return load_from_disk(train_path, train_images)

def init_test_images():
    return load_from_disk(test_path, test_images)

def get_test_set():
    for root, dirs, files in os.walk(train_path):
        for filename in files:
            # Open the image form working directory
            train_image = Image.open(train_path + filename)
            test_image = Image.open(test_path + filename)
            train_data = np.asarray(train_image)
            train_data = flatten(train_data)
            train_data = boolean_to_bipolar(train_data)
            test_data = np.asarray(test_image)
            test_data = flatten(test_data)
            test_data = boolean_to_bipolar(test_data)
            test_set.append([train_data, test_data])
    return test_set


def load_from_disk(path, images_array):
    for root, dirs, files in os.walk(path):
        for filename in files:
            # Open the image form working directory
            image = Image.open(path + filename)
            print("Loading " + path + filename + " Size:" + str(image.size))
            data = np.asarray(image)
            data = flatten(data)
            data = boolean_to_bipolar(data)
            images_array.append(data)
    return images_array
    
    
def get_pattern_length():
    return (128*128)

def flatten(m):
    flat_pattern = []
    for o in m:
        for p in o:
            flat_pattern.append(p)       
    return np.array(flat_pattern)
    
def boolean_to_binary(a):
    ca = [1 if x==True else 0 for x in a]
    return ca

def boolean_to_bipolar(a):
    ca = [1 if x==True else -1 for x in a]
    return ca
