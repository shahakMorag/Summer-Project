import cv2
from keras.applications import mobilenet

def rgb2hsv(image, use_imagenet_preprocess=True):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if use_imagenet_preprocess:
        return mobilenet.preprocess_input(image)
    else:
        return image


def rgb2hls(image, use_imagenet_preprocess=True):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    if use_imagenet_preprocess:
        return mobilenet.preprocess_input(image)
    else:
        return image
