import os
import cv2
from crop_utils import *

from keras.models import load_model


def get_images_location_list(images_location_file):
    with open(images_location_file) as f:
        return f.read().splitlines()


def create_file_name(start_location):
    filename = "_".join(start_location.split("\\")[-3:])
    path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\our_results"
    return os.path.join(path, filename)


def apply_on_all(save_path):
    model = load_model('../models/mobilenet/2018_08_26_23_18_1000_epochs_class_all.model')
    locations = get_images_location_list(
        "C:/Tomato_Classification_Project/Tomato_Classification_Project/Alon_misc/tomato_mark_AZ_20180803/mark"
        "/selected_file_list.txt")[215:220]
    images = segment_images(locations, model)
    for image, location in zip(images, locations):
        cv2.imwrite(create_file_name(location), image)


apply_on_all("C:\Tomato_Classification_Project\Tomato_Classification_Project\our_results")