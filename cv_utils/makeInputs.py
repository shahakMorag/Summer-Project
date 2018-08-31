from os import listdir, path
import numpy as np
from crop_utils import load_image


def get_pictures(pictures_directory):
    images = [load_image(path.join(pictures_directory, filename)) for filename in listdir(pictures_directory)]
    return images, len(images)


def make_inputs(root_path):
    print("Making imputs:")
    categories = ["bad_leaf", "fruit", "leaf", "other", "stem"]

    x, y = np.empty(0), []
    for category, i in zip(categories, range(len(categories))):
        pictures, pictures_count = get_pictures(path.join(root_path, category))
        x = np.concatenate((x, pictures))
        y += [i] * pictures_count

    return x, y
