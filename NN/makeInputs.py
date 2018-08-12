from PIL import Image
import glob
import numpy

limit = 100


def get_pictures(dir, limit):
    images = []
    res = []

    for filename in glob.glob(dir + '/*.png'):
        limit -= 1
        if limit == 0:
            break
        images.append(Image.open(filename))

    for x in images:
        res.append(numpy.array(x))

    return res


def make_inputs():
    X, Y = [], []
    X = get_pictures(
        "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf",
        limit)
    Y.append([[1, 0, 0, 0, 0]] * (limit - 1))
    X += get_pictures(
        "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/fruit",
        limit)
    Y.append([[0, 1, 0, 0, 0]] * (limit - 1))
    X += (get_pictures(
        "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/leaf",
        limit))
    Y.append([[0, 0, 1, 0, 0]] * (limit - 1))
    X += (get_pictures(
        "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/other",
        limit))
    Y.append([0, 0, 0, 1, 0] * (limit - 1))
    X += (get_pictures(
        "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/stem",
        limit))
    Y.append([[0, 0, 0, 0, 1]] * (limit - 1))
    return X, Y

