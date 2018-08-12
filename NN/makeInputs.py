from PIL import Image
import glob
import numpy

limit = 1000
path = "F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches"

def get_pictures(dir, limit):
    images = []
    res = []

    for filename in glob.glob(dir + '/*.png'):
        limit -= 1
        if limit == -1:
            break
        images.append(Image.open(filename))

    for x in images:
        res.append(numpy.array(x))

    return res


def make_inputs():
    X = get_pictures(path + "\patches_size_128_skip_32_categories_5/bad_leaf", limit)
    Y = [[1, 0, 0, 0, 0]] * limit
    X += get_pictures(path + "\patches_size_128_skip_32_categories_5/fruit", limit)
    Y += ([[0, 1, 0, 0, 0]] * limit)
    X += (get_pictures(path + "\patches_size_128_skip_32_categories_5/leaf", limit))
    Y += ([[0, 0, 1, 0, 0]] * limit)
    X += (get_pictures(path + "\patches_size_128_skip_32_categories_5/other", limit))
    Y += ([[0, 0, 0, 1, 0]] * limit)
    X += (get_pictures(path + "\patches_size_128_skip_32_categories_5/stem", limit))
    Y += ([[0, 0, 0, 0, 1]] * limit)
    return X, Y
