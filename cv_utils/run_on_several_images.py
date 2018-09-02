import os
import cv2
from crop_utils import *

from keras.models import load_model


def get_images_location_list(images_location_file):
    with open(images_location_file) as f:
        return f.read().splitlines()


def create_file_name(test_name, start_location):
    filename = test_name.replace(" ", "_")
    path = "../test/outputs"
    return os.path.join(path, filename + ".jpg")


def apply_on_all(test_name, model_path, step, radius):
    print("test name:", test_name)
    model = load_model(model_path)
    locations = ['../test\image transformations/IMG_5562.JPG']
    images = segment_images(locations, model, step=step, radius=radius)
    for image, location in zip(images, locations):
        cv2.imwrite(create_file_name(test_name ,location), image)


apply_on_all("RGB size 100 stride 13", '../models/2018_08_31_22_30_500_epochs_rgb_size_100.model', 13, 50)
apply_on_all("RGB size 100 stride 16", '../models/2018_08_31_22_30_500_epochs_rgb_size_100.model', 16, 50)
apply_on_all("RGB size 100 stride 20", '../models/2018_08_31_22_30_500_epochs_rgb_size_100.model', 20, 50)

apply_on_all("RGB size 128 stride 13", '../models/2018_09_01_0_4_500_epochs_rgb_size_128.model', 13, 64)
apply_on_all("RGB size 128 stride 16", '../models/2018_09_01_0_4_500_epochs_rgb_size_128.model', 16, 64)
apply_on_all("RGB size 128 stride 20", '../models/2018_09_01_0_4_500_epochs_rgb_size_128.model', 20, 64)

apply_on_all("RGB size 180 stride 13", '../models/2018_09_01_2_32_500_epochs_rgb_size_180.model', 13, 90)
apply_on_all("RGB size 180 stride 16", '../models/2018_09_01_2_32_500_epochs_rgb_size_180.model', 16, 90)
apply_on_all("RGB size 180 stride 20", '../models/2018_09_01_2_32_500_epochs_rgb_size_180.model', 20, 90)

'''
apply_on_all("HSV size 128 stride 13", '../models/2018_09_01_17_29_500_epochs_hsv_size_128.model', 13, 64)
apply_on_all("HSV size 128 stride 16", '../models/2018_09_01_17_29_500_epochs_hsv_size_128.model', 16, 64)
apply_on_all("HSV size 128 stride 20", '../models/2018_09_01_17_29_500_epochs_hsv_size_128.model', 20, 64)

apply_on_all("HLS size 128 stride 13", '../models/2018_09_02_2_7_500_epochs_hls_size_128.model', 13, 64)
apply_on_all("HLS size 128 stride 16", '../models/2018_09_02_2_7_500_epochs_hls_size_128.model', 16, 64)
apply_on_all("HLS size 128 stride 20", '../models/2018_09_02_2_7_500_epochs_hls_size_128.model', 20, 64)
'''