from cv_utils.crop_utils import *
from keras.models import load_model
import cv2

image_path = "../test/image transformations/IMG_5562.JPG"
model_path = "../models/mobilenet/2round/2018_08_29_22_17_500_epochs_round_1_3_classes.model"
original_image = load_image(image_path)
crops = create_crops(original_image)

fix_map = dict(zip(range(3), [1, 3, 4]))
raw_tags = apply_classification(crops, model_path)[0]
final_model_tags = [fix_map[raw_tag] for raw_tag in raw_tags]

''' ------------------------- second round ------------------------- '''

left_indexes = np.where(np.isin(final_model_tags, [0, 2, 3]))[0].tolist()
left_crops = crops[left_indexes]

# give path to the second round model
round_2_tags = apply_classification(left_crops,
                                    model_path="../models/mobilenet/2round/2018_08_30_0_41_500_epochs_round_2_3_classes.model")
fix_classes(final_model_tags, round_2_tags, left_indexes)

target_height, target_width = calc_dim(original_image)

reconstructed_image = keys2img(final_model_tags, target_height, target_width)[0]
resized_reconstructed_image = cv2.resize(reconstructed_image, None, fx=2.5, fy=2.5)
final_image = blend_two_images(resized_reconstructed_image, original_image, alpha=0.7)
cv2.imshow("window", final_image)
cv2.waitKey(0)
