import time

import cv2
from enc_dec_create_image import create_image
import argparse
from os import path
from crop_utils import keys2img


def get_video_properties(m_vidcap):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = m_vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = m_vidcap.get(cv2.CAP_PROP_FPS)

    vid_width = m_vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    vid_height = m_vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    length = int(m_vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    return int(vid_width),  int(vid_height), fps, length


def get_target_name(orig_path):
    p, name = path.split(orig_path)
    name, ext = path.splitext(name)
    return name


def prepare_batch(vid_height, vid_width, original_frames, segnmented_frames, alpha):
    res = []
    for seg_image, i in zip(segnmented_frames, range(len(segnmented_frames))):
        height, width = seg_image.shape[:2]
        seg_image = keys2img(seg_image.flatten(), height, width)[0]
        current_frame = original_frames[i]
        current_frame = current_frame[:vid_height - (vid_height % 372), :vid_width - (vid_width % 372)]
        current_frame = cv2.resize(current_frame, (vid_height, vid_width))
        matched_size_seg_frame = cv2.resize(seg_image, (vid_height, vid_width))
        overlayed_image = cv2.addWeighted(current_frame, alpha, matched_size_seg_frame, 1 - alpha, 0)
        resized_result = cv2.resize(overlayed_image, (vid_width, vid_height))
        res.append(resized_result)

    return res


def segment_movie(movie_path, save_dir, alpha, model_path):
    vidcap = cv2.VideoCapture(movie_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_width, vid_height, fps, length = get_video_properties(vidcap)
    y_mul_factor = 2
    x_mul_factor = 2
    vid_width *= x_mul_factor
    vid_height *= y_mul_factor

    save_name = path.join(*[save_dir, get_target_name(movie_path) + '.mp4'])
    video = cv2.VideoWriter(save_name, fourcc, fps, (vid_width, vid_height))
    count = 0
    batch_size = 60
    success = True
    frames = []
    limit = 100
    while success:
        limit -= 1
        success, vid_frame = vidcap.read()
        count += 1
        if success:
            vid_frame = cv2.resize(vid_frame, None, fx=x_mul_factor, fy=y_mul_factor)
            frames.append(vid_frame)

        if len(frames) == batch_size:
            print("vid_frame number: %d / %d" % (count, length))
            res = create_image(model_path=model_path, images=frames, batch_size=batch_size)

            res = prepare_batch(vid_height, vid_width, frames, res, alpha)
            for frame in res:
                video.write(frame)
            frames = []

        elif not success and len(frames):
            print("vid_frame number: %d / %d" % (count, length))
            res = create_image(model_path=model_path, images=frames)
            res = prepare_batch(vid_height, vid_width, frames, res, alpha)
            for frame in res:
                video.write(frame)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_path', required=True, help='Path to original video')
    parser.add_argument('-save_dir', required=True)
    parser.add_argument('-alpha', type=float, default=1)
    parser.add_argument('-model_path', required=True, help='path to encoder decoder model file')
    args = parser.parse_args()

    if path.exists(args.video_path) and path.exists(args.save_dir):
        segment_movie(args.video_path, args.save_dir, args.alpha, args.model_path)

    print("Video creation took %d seconds" % (time.time() - start_time,))
