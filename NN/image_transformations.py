import cv2
from keras.applications import mobilenet

def rgb2hsv(image, use_imagenet_preprocess=True):
    if use_imagenet_preprocess:
        image = mobilenet.preprocess_input(image)

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def rgb2hls(image, use_imagenet_preprocess=True):
    if use_imagenet_preprocess:
        image = mobilenet.preprocess_input(image)

    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def rgb2yuv(image, use_imagenet_preprocess=True):
    if use_imagenet_preprocess:
        image = mobilenet.preprocess_input(image)

    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
