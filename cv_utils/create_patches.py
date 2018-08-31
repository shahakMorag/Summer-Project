import glob
import os
from cv_utils.transformations import create_perspective_patches, create_rotated_patches
from cv_utils.crop_utils import *

base_path = "D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5"


def get_pictures(pictures_directory):
    for filename in glob.glob(os.path.join(pictures_directory, '*.png')):
        original_image = load_image(filename)
        perspective_images = create_perspective_patches(original_image)
        rotated_images = create_rotated_patches(original_image)
        filename = filename.replace('.png', '').replace('\\', '/')

        for img, i in zip(perspective_images, range(len(perspective_images))):
            cv2.imwrite(filename + '_per_' + repr(i) + '.png', img)

        for img, i in zip(rotated_images, range(len(rotated_images))):
            cv2.imwrite(filename + '_rot_' + repr(i) + '.png', img)


get_pictures(os.path.join(base_path, "bad_leaf"))
get_pictures(os.path.join(base_path, "fruit"))
get_pictures(os.path.join(base_path, "leaf"))
get_pictures(os.path.join(base_path, "other"))
get_pictures(os.path.join(base_path, "stem"))
