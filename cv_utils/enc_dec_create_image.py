import cv2
import keras
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from crop_utils import create_crops, calc_dim, keys2img


def create_image(model_path, image):
    output_size = 24

    crops = create_crops(image, 500, 500, 250, 250)
    height, width = calc_dim(image, 500, 500, 250, 250)
    target = np.zeros((height * output_size, width * output_size, 3), dtype=np.uint8)
    model = load_model(model_path)

    test_generator = ImageDataGenerator().flow(x=np.array(crops), batch_size=1, shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(crops),
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)
    predicts = np.array(predicts).reshape((-1, output_size*output_size,5))
    predicts = predicts.argmax(axis=2).reshape((-1,))
    predicts = keys2img(predicts, output_size, output_size, 60)
    predicts = predicts.reshape((height, width, output_size, output_size, 3))

    for y in range(height):
        for x in range(width):
            target[y*output_size:(y+1)*output_size, x * output_size:(x+1)*output_size] = predicts[y][x]

    return target


image = cv2.imread("../test/image transformations/IMG_5562.JPG", 1)
res = create_image("../models/encoder_decoder/m.model", image)
res = cv2.resize(res, None, fx=3, fy=3)
cv2.imshow("fds", res)
cv2.waitKey(10000)