import cv2
import glob
import numpy as np

limit = 8000
test = 100
path = "D:\Tomato_Classification_Project\Patches\Patches"


def make_one_hot(lst, num_classes):
    tmp = np.array(lst, dtype=np.int32)
    tmp2 = np.zeros((tmp.shape[0], num_classes), dtype=np.float32)
    tmp2[np.arange(tmp.shape[0]), tmp] = 1
    return tmp2

def get_pictures(dir, limit):
    images = []

    for filename in glob.glob(dir + '\*.png'):
        limit -= 1
        if limit == -1:
            break
        tmp = cv2.imread(filename, 1)

        images.append(tmp)

    return images


def make_inputs(num_classes, is_test=False):
    temp = get_pictures(path + "\\patches_size_128_skip_32_categories_5\\bad_leaf", limit + test)
    X = temp[:limit]
    X_test = temp[limit:]
    Y = [0] * limit
    Y_test = [0] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/fruit", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [1] * limit
    Y_test += [1] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/leaf", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [2] * limit
    Y_test += [2] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/other", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [3] * limit
    Y_test += [3] * test
    temp = get_pictures(path + "\patches_size_128_skip_32_categories_5/stem", limit + test)
    X += temp[:limit]
    X_test += temp[limit:]
    Y += [4] * limit
    Y_test += [4] * test

    if is_test:
        return np.array(X), make_one_hot(Y, num_classes), np.array(X_test), make_one_hot(Y_test, num_classes)

    return X, make_one_hot(Y, num_classes)
