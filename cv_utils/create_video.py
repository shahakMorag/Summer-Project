from os import path

import cv2

from enc_dec_create_image import create_image


def segment_movie(movie_path):
    images = []
    vidcap = cv2.VideoCapture(movie_path)
    success, image = vidcap.read()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = image.shape
    video = cv2.VideoWriter(r'D:/video.mp4', fourcc, 30.0, (width, height))
    print(image.shape)
    count = 0
    while success:
        images.append(cv2.resize(image, None, fx=2, fy=2))
        if len(images) == 60:
            print("image number:", count)
            res = create_image("../models/encoder_decoder/semantic_seg_2018_09_25_10_4.model", images, 60)
            for image, i in zip(res, range(len(res))):
                tmp_img = images[i]
                h, w = tmp_img.shape[:2]
                tmp_img = tmp_img[:h - (h % 372), :w - (w % 372)]
                video.write(cv2.resize(cv2.addWeighted(tmp_img, 0.3,
                                            cv2.resize(image, (tmp_img.shape[1], tmp_img.shape[0])), 0.7, 0), (width, height)))
            images = []
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    if len(images) > 0:
        res = create_image("../models/encoder_decoder/semantic_seg_2018_09_25_10_4.model", images)
        for image, i in zip(res, range(len(res))):
            tmp_img = images[i]
            h, w = tmp_img.shape[:2]
            tmp_img = tmp_img[:h - (h % 372), :w - (w % 372)]
            video.write(cv2.resize(cv2.addWeighted(tmp_img, 0.3,
                                                   cv2.resize(image, (tmp_img.shape[1], tmp_img.shape[0])), 0.7, 0),
                                   (width, height)))

    cv2.destroyAllWindows()
    video.release()


segment_movie(r'D:\test.mp4')
