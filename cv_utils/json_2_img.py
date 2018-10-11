import cv2
import json
from enc_dec_create_image import parse_color_mapping
from crop_utils import keys2img
import numpy as np
from os import path
import os
from tqdm import tqdm


def json_2_img(json_path, save_path):
    with open('encDecCreateImageConfig.json') as fp:
        args = json.load(fp)

    with open(json_path) as fp:
        json_img = np.array(json.load(fp))

    dir, name = path.split(json_path)
    save_path = path.join(save_path, name).replace("txt", "png")
    height, width = json_img.shape[:2]


    mapping = parse_color_mapping(args["mappingColors"])
    img = keys2img(json_img.flatten(), height, width, mapping=mapping)
    cv2.imwrite(save_path, img[0])


def create_if_nexists(dir):
    if not path.exists(dir):
        os.mkdir(dir)


def recurrent_run(src_dir, target_dir, function):
    print("Directory:", src_dir)
    for name in os.listdir(src_dir):
        full_path = path.join(src_dir, name)
        if path.isdir(full_path):
            new_target_dir = path.join(target_dir, name)
            create_if_nexists(new_target_dir)
            recurrent_run(full_path, new_target_dir, function)
        else:
            function(full_path, target_dir)


if __name__ == '__main__':

    with open('encDecCreateImageConfig.json') as fp:
        args = json.load(fp)

    base_dir = args["destPath"]
    json_dir = path.join(base_dir, "json")
    if not path.exists(json_dir):
        print("no base directory")
        exit(0)
    save_dir = path.join(base_dir, "colored")
    create_if_nexists(save_dir)

    recurrent_run(json_dir, save_dir, json_2_img)







