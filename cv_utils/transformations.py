import cv2
import numpy as np


def noBlackCrop(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    h, w = im.shape[:2]

    cols, rows = [], []

    for x in range(0, w, 1):
        tmp_col = [row[x] for row in imgray]
        first, last, y = 0, 0, 0

        while y < h and tmp_col[0, y] == [0, 0, 0]:
            print(tmp_col[0, y])
            y += 1

    cv2.imshow("sa", img)


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
    return cv2.warpPerspective(im, mat, (w, h))


im = cv2.imread('../test/image transformations/000035.png', 1)


def create_patches(img):
    #                  up-left, up-right, down-left, down-right
    # apply some perspective transformations
    pts1 = np.float32([[0, 0], [328, -200], [0, 128], [328, 328]])
    im_pers_1 = apply_perspective(im, pts1)

    pts2 = np.float32([[0, 0], [128, 0], [-200, 328], [328, 328]])
    im_pers_2 = apply_perspective(im, pts1)

    pts3 = np.float32([[-200, -200], [328, -200], [0, 128], [128, 128]])
    im_pers_3 = apply_perspective(im, pts1)

    pts4 = np.float32([[-200, -200], [128, 0], [-200, 328], [128, 128]])
    im_pers_4 = apply_perspective(im, pts1)

    # apply some rotations
    im_rot_1 = apply_rotation(im, 45, 1.4)
    im_rot_2 = apply_rotation(im, 90, 1)
    im_rot_3 = apply_rotation(im, 135, 1.4)
    im_rot_4 = apply_rotation(im, 180, 1)
    im_rot_5 = apply_rotation(im, 225, 1.4)
    im_rot_6 = apply_rotation(im, 270, 1)
    im_rot_7 = apply_rotation(im, 315, 1.4)

    img_pers = [im_pers_1, im_pers_2, im_pers_3, im_pers_4]
    img_rot = [im_rot_1, im_rot_2, im_rot_3, im_rot_4, im_rot_5, im_rot_6, im_rot_7]

    return img_pers + img_rot


im3 = adjust_gamma(im, 0.3)
im4 = adjust_gamma(im, 0.4)
im5 = adjust_gamma(im, 0.5)
im6 = adjust_gamma(im, 0.6)
im7 = adjust_gamma(im, 0.7)

cv2.waitKey(0)
