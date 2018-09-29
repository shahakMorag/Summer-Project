import numpy as np
import cv2
import json
from os import listdir, path
from tqdm import tqdm

# 0 - bad leaf - blue    - [255, 0, 0]
# 1 - fruit    - red     - [35,28,229]
# 2 - leaf     - green   - [0, 255, 0]
# 3 - other    - brown   - [0,255,239]
# 4 - stem - dark green  - [16,64,4]
values = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]

keys = ["255 0 0", "35 28 229", "0 255 0", "0 255 239", "16 64 4", "0 0 0"]

colors_to_tags = dict(zip(keys, values))


# [0, 0, 255] - red
# [0, 255, 0] - green
# [255, 0, 0] - blue


def image_to_onehots(image):
    height, width = image.shape[:2]
    return [[colors_to_tags.__getitem__(" ".join(image[y][x].astype(np.str))) for x in range(width)] for y in range(height)]


def create_store_name(image_name):
    return path.join("C:\Tomato_Classification_Project\Tomato_Classification_Project\encoder_decoder_train_set",
                     image_name.replace(".png", ".txt"))


def convert_images_to_onehots(directory):
    for image_name in tqdm(listdir(directory)):
        image = cv2.imread(path.join(directory, image_name), 1)
        image = image_to_onehots(image)
        with open(create_store_name(image_name), "w") as store_file:
            store_file.write(json.dumps(image))

if __name__ == '__main__':
    convert_images_to_onehots("C:\Tomato_Classification_Project\Tomato_Classification_Project/targets_encdec")