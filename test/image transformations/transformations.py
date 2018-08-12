from PIL import Image, ImageChops, ImageFilter, ImageTransform, ImageOps
import numpy as np
import math


im = Image.open("Tomato.jpg")
im = im.resize((128, 128))


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
    image = image.convert("RGB")
    A = np.random.exponential(1/7, (3, 3))
    b = np.random.rand(3)
    return change_color_space(image, lambda x: add_tuples(rgb_transform(A, x), b))


# change_color_space(im, rgb2uyv).save("tomato_in_yuv_colorspace.jpg")

#change_color_space(im, rgb2tomato).show()
'''random_color_space(im).show()
random_color_space(im).show()
random_color_space(im).show()
random_color_space(im).show()
random_color_space(im).show()'''

# change_color_space(im, rgb_sha).save("tomato_in_random_colorspace.jpg")


def move_to_middle(image):
    n_image = image.convert('RGB')
    return n_image.transform((n_image.size[0] * 2, n_image.size[1] * 2), Image.AFFINE,(1,0,-n_image.size[0]/2,0,1,-n_image.size[1]/2))

def affine_transformation1(image):
    n_image = move_to_middle(image)
    return ImageOps.crop(n_image.transform(n_image.size, Image.AFFINE, (-0.75, 0.25,192,0.05,0.8,10)), 64)


def affine_transformation2(image):
    n_image = move_to_middle(image)
    return ImageOps.crop(n_image.transform(n_image.size, Image.AFFINE, (0.9, -0.1,25,-0.1,0.5,80)), 64)


def affine_transformation3(image):
    n_image = move_to_middle(image)
    return ImageOps.crop(n_image.transform(n_image.size, Image.AFFINE, (-0.8, .2,205,0.4,-0.5,140)), 64)


affine_transformation1(im)#.resize((128, 128)).show()
affine_transformation2(im)#.resize((128, 128)).show()
affine_transformation3(im).show()#.resize((128, 128)).show()
