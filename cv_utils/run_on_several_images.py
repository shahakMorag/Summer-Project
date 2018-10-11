import argparse
import json
import os
from os import path
from crop_utils import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model


def get_images_location_list(images_location_file):
    with open(images_location_file) as f:
        return f.read().splitlines()


def create_file_name(start_location, save_dir):
    # filename = "_".join(start_location.split("\\")[-1])
    head, tail = path.split(start_location)
    filename = tail.lower().replace("png", "txt").replace('jpg', 'txt').replace('jpeg', 'txt')
    return os.path.join(save_dir, filename)


def apply_on_all(model_path, locations, save_dir, step, radius, num_classes, jump):
    for i in range(0, len(locations), jump):
        model = load_model(model_path)
        tmp_locations = locations[i:i + jump]
        images = segment_images(tmp_locations, model, num_classes, step=step, radius=radius)

        for image, location in zip(images, tmp_locations):
            with open(create_file_name(location, save_dir), 'w') as f:
                f.write(json.dumps(image))

        model = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_image_number', required=True, type=int)
    parser.add_argument('-num_classes', required=True, type=int)
    parser.add_argument('-jump', required=True, type=int)
    parser.add_argument('-model_path', required=True)
    parser.add_argument('-patches_path', required=True)
    parser.add_argument('-save_dir', required=True)
    args = parser.parse_args()
    images_paths = [path.join(args.patches_path, str(i) + '.png') for i in
                  range(args.start_image_number, args.start_image_number + args.jump)]

    apply_on_all(args.model_path, images_paths, args.save_dir, 16, 64, args.num_classes, args.jump)
