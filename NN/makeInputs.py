import cv2
import glob
import numpy as np


def make_one_hot(lst, num_classes):
    tmp = np.array(lst, dtype=np.int32)
    tmp2 = np.zeros((tmp.shape[0], num_classes), dtype=np.float32)
    tmp2[np.arange(tmp.shape[0]), tmp] = 1
    return tmp2


def get_pictures(dir):
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


def make_inputs(path, num_classes):
    print("Making imputs:")
    X = get_pictures(path + "\\bad_leaf", start, limit )
    Y = [0] * limit

    X = np.concatenate((X, get_pictures(path + "\\fruit", start, limit )))
    Y += [1] * limit

    X = np.concatenate((X,get_pictures(path + "\\leaf", start, limit )))
    Y += [2] * limit

    X = np.concatenate((X, get_pictures(path + "\\other", start, limit )))
    Y += [3] * limit

    X = np.concatenate((X,get_pictures(path + "\\stem", start, limit )))
    Y += [4] * limit

    print("Finished making inputs")

    return X, make_one_hot(Y, num_classes)
