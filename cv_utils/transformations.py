import cv2
import numpy as np
from NN.makeInputs import make_inputs

path = path = "D:\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5"


def get_relative_brightness(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    expect = 0
    i = 0
    for x in hist:
        expect += i * x[0]
        i += 1

    return expect / (h * w)


def correct_gamma(img):
    br = get_relative_brightness(img)
    n_img = img

    if br > 110:
        while br > 110:
            br = get_relative_brightness(n_img)
            n_img = adjust_gamma(n_img, 0.9)

    else:
        while br < 110:
            br = get_relative_brightness(n_img)
            n_img = adjust_gamma(n_img, 1.1)

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
    pts1 = np.float32([[0, 0], [328, -200], [0, 128], [328, 328]])
    im_pers_1 = apply_perspective(img, pts1)

    pts2 = np.float32([[0, 0], [128, 0], [-200, 328], [328, 328]])
    im_pers_2 = apply_perspective(img, pts1)

    pts3 = np.float32([[-200, -200], [328, -200], [0, 128], [128, 128]])
    im_pers_3 = apply_perspective(img, pts1)

    pts4 = np.float32([[-200, -200], [128, 0], [-200, 328], [128, 128]])
    im_pers_4 = apply_perspective(img, pts1)

    return [im_pers_1, im_pers_2, im_pers_3, im_pers_4]


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