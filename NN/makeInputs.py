import cv2
import glob
import numpy as np

path = "C:\Tomato_Classification_Project_5_iter\Patches\Patches"


def make_one_hot(lst, num_classes):
    tmp = np.array(lst, dtype=np.int32)
    tmp2 = np.zeros((tmp.shape[0], num_classes), dtype=np.float32)
    tmp2[np.arange(tmp.shape[0]), tmp] = 1
    return tmp2


def get_pictures(dir, start, limit):
    images = np.empty([limit,128,128,3], dtype=np.uint8)
    i = 0

    while limit > 0:
        for filename in glob.glob(dir + '\*.png'):
            if start > 0:
                start -= 1
                continue

            images[i] = cv2.imread(filename, 1)
            i += 1
            limit -= 1
            if limit == 0:
                break

    return images


def make_inputs(start, limit, num_classes):
    print("Making imputs:")
    print("Start: " + repr(start) + ", limit: " + repr(limit) + ", classes: " + repr(num_classes))
    X = get_pictures(path + "\\patches_size_128_skip_32_categories_5\\bad_leaf", start, limit )
    Y = [0] * limit

    X = np.concatenate((X, get_pictures(path + "\\patches_size_128_skip_32_categories_5\\fruit", start, limit )))
    Y += [1] * limit

    X = np.concatenate((X,get_pictures(path + "\\patches_size_128_skip_32_categories_5\\leaf", start, limit )))
    Y += [2] * limit

    X = np.concatenate((X, get_pictures(path + "\\patches_size_128_skip_32_categories_5\\other", start, limit )))
    Y += [3] * limit

    X = np.concatenate((X,get_pictures(path + "\\patches_size_128_skip_32_categories_5\\stem", start, limit )))
    Y += [4] * limit

    print("Finished making inputs")

    return X, make_one_hot(Y, num_classes)
