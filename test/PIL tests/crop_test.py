from PIL import Image, ImageFilter, ImageChops
import numpy as np
import random

im = Image.open("../image transformations/Tomato.jpg")
im = im.resize((500, 500))
size = im.size

step = 120
crop_x = 150
crop_y = 150


def createCrops(im, step_x, step_y, crop_x, crop_y):
    res = []
    for up in range(0, im.size[1] - crop_y, step_y):
        for left in range(0, im.size[0] - crop_x, step_x):
            cropped = im.crop((left, up, left + crop_x, up + crop_y))
            res.append(cropped)

    return res


def crops_show(im_list):
    for crop in im_list:
        crop.show()


list = createCrops(im, step, step, crop_x, crop_y)
list[0].save("Crop0.jpg")
list[2].save("Crop1.jpg")
list[3].save("Crop2.jpg")
list[8].save("Crop3.jpg")
