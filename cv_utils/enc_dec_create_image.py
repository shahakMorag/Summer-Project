import cv2
import keras
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from crop_utils import create_crops, calc_dim, keys2img


def create_image(model_path, images):
    output_size = 24

    crops = [create_crops(image, 372, 372, 372 // 2, 372 // 2) for image in images]
    crops = np.array(crops).reshape((-1, 372, 372, 3))
    height, width = calc_dim(images[0], 372, 372, 372 // 2, 372 // 2)
    target = np.zeros((len(images), height * output_size, width * output_size, 3), dtype=np.uint8)
    model = load_model(model_path)

    test_generator = ImageDataGenerator().flow(x=np.array(crops), batch_size=1, shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(crops),
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)

    predicts = np.array(predicts).reshape((-1, output_size * output_size, 5))
    predicts = predicts.argmax(axis=-1).reshape((-1,))
    predicts = keys2img(predicts, output_size, output_size, height * width)
    predicts = predicts.reshape((height, width, output_size, output_size, 3))

    for image_index in range(len(images)):
        for y in range(height):
            for x in range(width):
                target[image_index, y * output_size:(y + 1) * output_size, x * output_size:(x + 1) * output_size] = predicts[y][x]

    return target


image = cv2.imread("../test/image transformations/IMG_0781.JPG", 1)
image1 = cv2.imread("../test/image transformations/IMG_0781.JPG", 1)
cv2.imshow("t", cv2.resize(image, None, fx=0.2, fy=0.2))
cv2.waitKey(2000)
res = create_image("../models/encoder_decoder/semantic_seg_2018_09_21_7_51.model", [image])
cv2.imwrite("C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data/test/enc_dec_5562.png", res)

cv2.imshow("fds", res[0])
cv2.waitKey(10000)
