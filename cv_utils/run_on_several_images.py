import os
import cv2
from crop_utils import *
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

jump = 40

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True, type=int)
args = parser.parse_args()
patches_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\size_500_stride_500/'
image_path = [patches_path + str(i) + '.png' for i in range(args.p,args.p + jump)]

from keras.models import load_model


def get_images_location_list(images_location_file):
    with open(images_location_file) as f:
        return f.read().splitlines()


def create_file_name(start_location):
    # filename = "_".join(start_location.split("\\")[-1])
    filename = start_location.split("/")[-1]
    path = "C:\Tomato_Classification_Project\Tomato_Classification_Project/targets_encdec"
    return os.path.join(path, filename)


def apply_on_all(model_path, step, radius):
    # locations = get_images_location_list("C:\Tomato_Classification_Project\Tomato_Classification_Project\Alon_misc/tomato_mark_AZ_20180803\mark/selected_file_list.txt")
    images = []
    locations = image_path
    for i in range(0, len(locations), jump):
        model = load_model(model_path)
        last = min(i+jump, len(locations))
        tmp_locations = locations[i:last]
        images = segment_images(tmp_locations, model, step=step, radius=radius)

        for image, location in zip(images, tmp_locations):
            cv2.imwrite(create_file_name(location), image)

        model = None


apply_on_all('../models/mobilenet/all_models/2018_09_01_0_4_500_epochs_rgb_size_128.model', 16, 64)
