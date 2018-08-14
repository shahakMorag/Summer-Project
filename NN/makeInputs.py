from PIL import Image
import glob
import numpy

limit = 2000
test = 100
path = "C:\Tomato_Classification_Project\Patches\Patches"


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


def make_inputs(is_test=False):
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/bad_leaf", limit + test)
    X = temp[:limit]
    X_test = temp[limit:]
    Y = [[1, 0, 0, 0, 0]] * limit
    Y_test = [[1, 0, 0, 0, 0]] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/fruit", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [[0, 1, 0, 0, 0]] * limit
    Y_test += [[0, 1, 0, 0, 0]] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/leaf", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [[0, 0, 1, 0, 0]] * limit
    Y_test += [[0, 0, 1, 0, 0]] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/other", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [[0, 0, 0, 1, 0]] * limit
    Y_test += [[0, 0, 0, 1, 0]] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/stem", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [[0, 0, 0, 0, 1]] * limit
    Y_test += [[0, 0, 0, 0, 1]] * test

    if is_test:
        return X, Y, X_test, Y_test

    return X, Y
