import cv2
from enc_dec_create_image import create_image
import argparse
from os import path


def segment_movie(movie_path, save_dir, alpha, model_path):
    images = []
    vidcap = cv2.VideoCapture(movie_path)
    success, image = vidcap.read()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = image.shape
    video = cv2.VideoWriter(save_dir, fourcc, 30.0, (width, height))
    print(image.shape)
    count = 0
    while success:
        images.append(cv2.resize(image, None, fx=2, fy=2))
        if len(images) == 60:
            print("image number:", count)
            res = create_image(model_path, images, 60)
            for image, i in zip(res, range(len(res))):
                tmp_img = images[i]
                h, w = tmp_img.shape[:2]
                tmp_img = tmp_img[:h - (h % 372), :w - (w % 372)]
                video.write(cv2.resize(cv2.addWeighted(tmp_img, alpha,
                                        cv2.resize(image, (tmp_img.shape[1], tmp_img.shape[0])), 1-alpha, 0),
                                       (width, height)))
            images = []
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    if len(images) > 0:
        res = create_image(model_path, images)
        for image, i in zip(res, range(len(res))):
            tmp_img = images[i]
            h, w = tmp_img.shape[:2]
            tmp_img = tmp_img[:h - (h % 372), :w - (w % 372)]
            video.write(cv2.resize(cv2.addWeighted(tmp_img, alpha,
                                    cv2.resize(image, (tmp_img.shape[1], tmp_img.shape[0])), 1-alpha, 0), (width, height)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_path', required=True, help='Path to original video')
    parser.add_argument('-save_dir', required=True)
    parser.add_argument('-alpha', type=float, default=1)
    parser.add_argument('-model_path', required=True, help='path to encoder decoder model file')
    args = parser.parse_args()

    if path.exists(args.video_path) and path.exists(args.save_dir):
        segment_movie(args.video_path, args.save_dir, args.alpha, args.model_path)
