from crop_utils import *
from keras.models import load_model
import cv2
import time
import numpy as np

image_path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\Data\Demonstration_greenhouse_tomato_Hazera\Demonstration_greenhouse_tomato_Hazera\Pink\HTP_11/IMG_0236.JPG"
model_path = "../models/mobilenet/all_models/2018_09_01_13_13_500_epochs_rgb_128_round_1.model"
original_image = load_image(image_path)
crops = np.array(create_crops(original_image))

fix_map = dict(zip(range(3), [1, 3, 4]))
start_time = time.time()
raw_tags = apply_classification(crops, model_path, fix_function=lambda x: fix_map[x])

''' ------------------------- second round ------------------------- '''

left_indexes = np.where(np.isin(raw_tags, [3]))[0].tolist()
left_crops = crops[left_indexes]

# give path to the second round model
round_2_tags = apply_classification(left_crops,
                                    model_path="../models/mobilenet/all_models/2018_09_01_14_59_500_epochs_rgb_128_round_2.model")
print("total time took:", time.time()-start_time, "seconds")
fix_classes(raw_tags, round_2_tags, left_indexes)

target_height, target_width = calc_dim(original_image)

reconstructed_image = keys2img(raw_tags, target_height, target_width)[0]
resized_reconstructed_image = reconstructed_image
final_image = blend_two_images(resized_reconstructed_image, original_image, alpha=1)
# cv2.imshow("window", final_image)
# cv2.waitKey(0)
cv2.imwrite("C:\Tomato_Classification_Project\Tomato_Classification_Project/temp/2round.jpg", final_image)