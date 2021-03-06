from os import path

import cv2
import numpy as np
import time

step = 16
radius_x = 64
radius_y = 64


def create_crops(image, step_x=step, step_y=step, radius_x=radius_x, radius_y=radius_y):
    height, width = image.shape[:2]
    radius_x = int(radius_x)
    radius_y = int(radius_y)
    crops = [image[y_mid - radius_y:y_mid + radius_y, x_mid - radius_x:x_mid + radius_x]
            for y_mid in range(radius_y, height - radius_y, step_y)
            for x_mid in range(radius_x, width - radius_x, step_x)]
    return crops


def calc_dim(image, step_x=step, step_y=step, radius_x=radius_x, radius_y=radius_y):
    height, width = image.shape[:2]
    return len(range(radius_y, height - radius_y, step_y)), len(range(radius_x, width - radius_x, step_x))


def apply_classification(images_list,
                         model_path='../models/mobilenet/2018_08_27_21_58_5_epochs_leaf_other.model',
                         model=None,
                         fix_function=None):
    import keras
    from image_transformations import rgb2hsv, rgb2hls
    from keras.models import load_model
    from keras_preprocessing.image import ImageDataGenerator
    import keras.applications
    start_time = time.time()
    print("Applying classification...")

    # if the user doesn't pass a model
    if model is None:
        model = load_model(model_path)

    # keras.applications.mobilenet.preprocess_input

    test_generator = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) \
        .flow(x=np.array(images_list),
              batch_size=1,
              shuffle=False)

    predicts = model.predict_generator(test_generator,
                                       steps=len(images_list),
                                       verbose=1,
                                       workers=8,
                                       use_multiprocessing=False)

    tags = predicts.argmax(axis=1)
    print("Classification took", repr(time.time() - start_time), "seconds")
    if fix_function is None:
        return np.array(tags)

    return np.array([fix_function(tag) for tag in tags])


def fix_classes(original_tags, fixed_tags, index_to_fix):
    maps = dict(zip([0, 1, 2], [0, 2, 3]))

    for i in range(len(fixed_tags)):
        original_tags[index_to_fix[i]] = maps.__getitem__(fixed_tags[i])


# We assume that the images in the same size
def blend_two_images(neural_net_image, original_image, radius_x=radius_x, radius_y=radius_y, alpha=1.0):
    original_image = original_image[radius_y:-radius_y, radius_x:-radius_x]
    original_image = cv2.resize(original_image, tuple(neural_net_image.shape[1::-1]))

    beta = (1 - alpha)
    dst = cv2.addWeighted(neural_net_image, alpha, original_image, beta, 0.0)
    return dst


# 0 - bad leaf - blue    - [255, 0, 0]
# 1 - fruit    - red     - [35,28,229]
# 2 - leaf     - green   - [0, 255, 0]
# 3 - other    - brown   - [0,255,239]
# 4 - stem - dark green  - [16,64,4]

# [0, 0, 255] - red
# [0, 255, 0] - green
# [255, 0, 0] - blue

default_colors = np.array([[255, 0, 0],
                       [35, 28, 229],
                       [0, 255, 0],
                       [0, 255, 239],
                       [16, 64, 4]]).astype(np.uint8)
default_map = dict(zip(range(5), default_colors))


def keys2img(tags, height, width, num_images=1, mapping=default_map):
    res = [mapping.get(tag) for tag in tags]
    return np.reshape(res, (num_images, int(height), int(width), 3))


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path, 1), cv2.COLOR_BGR2RGB)


def segment_images(image_location_list, model, num_classes, step=step, radius=radius_x):
    import keras
    scale_after = 1

    num_images = len(image_location_list)
    raw_images = [load_image(image_location) for image_location in image_location_list]

    new_height, new_width = calc_dim(raw_images[0], step, step, radius, radius)

    crops_list = [np.array(create_crops(image, step, step, radius, radius)) for image in raw_images]
    crops_list = np.array(crops_list).reshape((-1, radius * 2, radius * 2, 3))
    classified = apply_classification(crops_list, model=model)
    classified = classified.reshape((num_images, new_height, new_width, 1))
    return np.array(keras.utils.to_categorical(classified, num_classes=num_classes), dtype=np.int32).tolist()
    # recovered_image = keys2img(classified, new_height, new_width, num_images)
    # return [blend_two_images(recovered_image[i], raw_images[i], radius, radius, alpha=1)
    #         for i in range(len(recovered_image))]


if __name__ == '__main__':
    from keras.models import load_model
    image_locations = [r"D:/cucumbers_project/cucumber_pics/2018_06_04_16_15_segmentation_task_26_nodes_on_stem_cucumber_BH/IMG_2085.JPG"]
    img = np.array(segment_images(image_locations,
                   load_model(r"C:\Users\eitan.k\PycharmProjects\Summer-Project\models\smaller_mobilenet/2018_09_30_0_54_500_epochs_Cucumber.model"),
                   num_classes=5))[0].argmax(axis=-1)
    height, width = img.shape[:2]
    color_img = keys2img(img.flatten(), height, width)
    cv2.imshow("", color_img[0])
    cv2.waitKey(0)
