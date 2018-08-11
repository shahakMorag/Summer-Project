from PIL import Image, ImageChops, ImageFilter
import numpy as np
import math

im = Image.open("Tomato.jpg")


def float_to_short(f):
    return math.ceil(f) % 256


def rgb_transform(A, rgb):
    r, g, b = rgb
    c = A @ np.array([r, g, b])
    r, g, b = float_to_short(c[0]), float_to_short(c[1]), float_to_short(c[2])
    return r, g, b


def rgb_sha(rgb):
    A = np.random.randn(3, 3)

    return rgb_transform(A, rgb)


def rgb2tomato(rgb):
    A = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

    return rgb_transform(A, rgb)

def rgb2uyv(rgb):
    A = np.array([[-0.1471, -0.2889, 0.4360],
                  [0.2990, 0.5870, 0.1140],
                  [0.6150, -0.5150, -0.1000]])

    return rgb_transform(A, rgb)


def change_color_space(image, func):
    h, w = image.size
    res = Image.new("RGB", image.size, (255, 255, 255))

    for i in range(h):
        for j in range(w):
            res.putpixel((i, j), func(image.getpixel((i, j))))

    return res


def add_tuples(a, b):
    return float_to_short(a[0] + b[0]), float_to_short(a[1] + b[1]), float_to_short(a[2] + b[2])


def random_color_space(image):
    A = np.random.exponential(1/4, (3, 3))
    b = np.random.rand(3)
    return change_color_space(image, lambda x: add_tuples(rgb_transform(A, x), b))


# change_color_space(im, rgb2uyv).save("tomato_in_yuv_colorspace.jpg")

change_color_space(im, rgb2tomato).show()
random_color_space(im).show()

# change_color_space(im, rgb_sha).save("tomato_in_random_colorspace.jpg")
