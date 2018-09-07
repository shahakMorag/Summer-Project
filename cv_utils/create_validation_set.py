from os import path, listdir, mkdir
import shutil
import random


def partially_move_images(source_path, destination_path, part):
    files_names = [name for name in listdir(source_path) if path.isfile(path.join(source_path, name))]
    sample_files = random.sample(files_names, int(len(files_names) * part))

    if not path.exists(destination_path):
        mkdir(destination_path)

    for name in sample_files:
        shutil.move(path.join(source_path, name), path.join(destination_path, name))


def move_sub_dirs(source_root, destination_root, part, swap=False):
    if not path.exists(destination_root):
        mkdir(destination_root)

    if swap:
        source_root, destination_root = destination_root, source_root

    partially_move_images(path.join(source_root, 'bad_leaf'), path.join(destination_root, 'bad_leaf'), part)
    partially_move_images(path.join(source_root, 'fruit'), path.join(destination_root, 'fruit'), part)
    partially_move_images(path.join(source_root, 'leaf'), path.join(destination_root, 'leaf'), part)
    partially_move_images(path.join(source_root, 'other'), path.join(destination_root, 'other'), part)
    partially_move_images(path.join(source_root, 'stem'), path.join(destination_root, 'stem'), part)


part = 0.2
source_path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5"
destination_path = "C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_128"
move_sub_dirs(source_path, destination_path, part, swap=False)
