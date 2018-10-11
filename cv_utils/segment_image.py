import argparse
import cv2
import json
from os import path, mkdir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_path', required=True)
    parser.add_argument('-color', type=bool)
    args = parser.parse_args()

    image_path = args.image_path
    original_image = cv2.imread(image_path, 1)
    if original_image is None:
        print("Failed to load image from", image_path)
        exit(0)

