from crop_utils import create_crops
from os import listdir, path
import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import argparse


def make_all_dirs(src, dest, size=500, stride=500):
    crop_number = 0
    for file in tqdm(glob.iglob(path.join(src, "**/*.JPG"), recursive=True)):
        image = cv2.imread(path.join(src, file), 1)
        crops_500 = create_crops(image=image,
                                 step_x=stride,
                                 step_y=stride,
                                 radius_x=size / 2,
                                 radius_y=size / 2)

        crops_372 = [crop[64:size - 64, 64:size - 64] for crop in crops_500]

        path_500 = path.join(dest, 'crops_500')
        path_372 = path.join(dest, 'crops_372')

        if not path.exists(dest): os.mkdir(dest)
        if not path.exists(path_500): os.mkdir(path_500)
        if not path.exists(path_372): os.mkdir(path_372)

        for crop_500, crop_372 in zip(crops_500, crops_372):
            file_name = str(crop_number) + ".png"
            cv2.imwrite(path.join(path_500, file_name), crop_500)
            cv2.imwrite(path.join(path_372, file_name), crop_372)
            crop_number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_path', required=True)
    parser.add_argument('-save_path', required=True)
    args = parser.parse_args()

    make_all_dirs(args.images_path, args.save_path)
