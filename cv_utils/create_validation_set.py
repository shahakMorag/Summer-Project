import os
from os import path
import shutil
import random


def partially_move_images(source_path, destination_path, part):
    files_names = [name for name in os.listdir(source_path) if path.isfile(os.path.join(source_path, name))]
    sample_files = random.sample(files_names, len(files_names) * part)

    for name in sample_files:
        shutil.move(path.join(source_path, name), path.join(destination_path, name))


def move_all(source_path, destination_path):
    partially_move_images(source_path, destination_path, 1)


def move_sub_dirs(source_root, destination_root, part):
    partially_move_images(path.join(source_root, 'bad_leaf'), path.join(destination_root, 'bad_leaf'), part)
    partially_move_images(path.join(source_root, 'fruit'), path.join(destination_root, 'fruit'), part)
    partially_move_images(path.join(source_root, 'leaf'), path.join(destination_root, 'leaf'), part)
    partially_move_images(path.join(source_root, 'other'), path.join(destination_root, 'other'), part)
    partially_move_images(path.join(source_root, 'stem'), path.join(destination_root, 'stem'), part)


part = 0.08

source_path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5"
destination_path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation"

move_sub_dirs(source_path, destination_path, part)
