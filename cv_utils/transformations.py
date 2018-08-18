import cv2
import numpy as np
from NN.makeInputs import make_inputs

path = path = "D:\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5"


def get_relative_brightness(img, channels):
    if channels == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    elif channels == 1:
        return img.mean(axis=None)

    h, w = img.shape[:2]
    expect = 0
    i = 0
    for x in hist:
        expect += i * x[0]
        i += 1

    return expect / (h * w)


def correct_gamma(img):
    br = get_relative_brightness(img, 3)
    n_img = img
    threshold = 7

    if br > 110:
        while br > 110 and threshold > 0:
            br = get_relative_brightness(n_img, 3)
            n_img = adjust_gamma(n_img, 0.9)
            threshold -= 1

    else:
        while br < 110 and threshold > 0:
            br = get_relative_brightness(n_img, 3)
            n_img = adjust_gamma(n_img, 2)
            threshold -= 1

    return n_img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def apply_rotation(img, deg, scale):
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((h / 2, w / 2), deg, scale)
    return cv2.warpAffine(img, mat, (w, h))


def apply_perspective(img, dest):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])
    mat = cv2.getPerspectiveTransform(pts1, dest)
    return cv2.warpPerspective(img, mat, (w, h))


def create_perspective_patches(img):
    #                  up-left, up-right, down-left, down-right
    # apply some perspective transformations
    param = 30

    # trapezoids
    pts1 = np.float32([[0, 0], [128 + param, -param], [0, 128], [128 + param, 128 + param]])
    im_pers_1 = apply_perspective(img, pts1)

    pts2 = np.float32([[0, 0], [128, 0], [-param, 128 + param], [128 + param, 128 + param]])
    im_pers_2 = apply_perspective(img, pts2)

    pts3 = np.float32([[-param, -param], [128 + param, -param], [0, 128], [128, 128]])
    im_pers_3 = apply_perspective(img, pts3)

    pts4 = np.float32([[-param, -param], [128, 0], [-param, 128 + param], [128, 128]])
    im_pers_4 = apply_perspective(img, pts4)

    # merubaim
    pts5 = np.float32([[0, 0], [128, -param], [-param, 128], [128 + 1.2 * param, 128 + 1.2 * param]])
    im_pers_5 = apply_perspective(img, pts5)

    pts6 = np.float32([[0, - param], [128, 0], [-1.2 * param, 128 + 1.2 * param], [128 + 1.2 * param, 128]])
    im_pers_6 = apply_perspective(img, pts6)

    pts7 = np.float32([[-1.2 * param, -1.2 * param], [128 + param, 0], [0, 128 + param], [128, 128]])
    im_pers_7 = apply_perspective(img, pts7)

    pts8 = np.float32([[-param, 0], [128 + 1.2 * param, -1.2 * param], [0, 128], [128, 128 + param]])
    im_pers_8 = apply_perspective(img, pts8)

    return [im_pers_1, im_pers_2, im_pers_3, im_pers_4, im_pers_5, im_pers_6, im_pers_7, im_pers_8]


def create_rotated_patches(img):
    # apply some rotations
    im_rot_1 = apply_rotation(img, 45, 1.4)
    im_rot_2 = apply_rotation(img, 90, 1)
    im_rot_3 = apply_rotation(img, 135, 1.4)
    im_rot_4 = apply_rotation(img, 180, 1)
    im_rot_5 = apply_rotation(img, 225, 1.4)
    im_rot_6 = apply_rotation(img, 270, 1)
    im_rot_7 = apply_rotation(img, 315, 1.4)

    return [im_rot_1, im_rot_2, im_rot_3, im_rot_4, im_rot_5, im_rot_6, im_rot_7]


def sharpen(im):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(im, -1, kernel)
