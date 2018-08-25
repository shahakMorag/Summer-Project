import os
import shutil
import random

def create_precense(source_path, dest_path, part):
    count = len([name for name in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, name))])
    target_count = count * part
    moved = 0

    while moved < target_count:
        wait = random.randint(0, count - moved - 1)
        name = None
        for filename in os.listdir(source_path):
            if wait > 0:
                wait -= 1
                continue

            name = filename
            break

        dest_name = dest_path + '//' + filename
        shutil.move(source_path + '//' + filename, dest_name)
        moved += 1



def create_quantity(dest_path, source_path, amount):
    count = len([name for name in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, name))])
    target_count = amount
    moved = 0

    while moved < target_count:
        wait = random.randint(0, count - moved - 1)
        name = None
        for filename in os.listdir(source_path):
            if wait > 0:
                wait -= 1
                continue

            name = filename
            break

        dest_name = dest_path + '//' + filename
        shutil.move(source_path + '//' + filename, dest_name)
        moved += 1


# Moves all files from src path dir to destination path dir
def move_all(source_path, dest_path, part=1):
    create_precense(source_path, dest_path, part)


def move_sub_dirs(src_root, dst_root, part):
    move_all(src_root + '/' + 'bad_leaf', dst_root + '/' + 'bad_leaf', part)
    move_all(src_root + '/' + 'fruit', dst_root + '/' + 'fruit', part)
    move_all(src_root + '/' + 'leaf', dst_root + '/' + 'leaf', part)
    move_all(src_root + '/' + 'other', dst_root + '/' + 'other', part)
    move_all(src_root + '/' + 'stem', dst_root + '/' + 'stem', part)

'''
move_all('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/bad_leaf',
         'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf')

move_all('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/fruit',
         'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/fruit')

move_all('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/leaf',
         'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/leaf')

move_all('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/other',
         'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/other')

move_all('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/stem',
         'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/stem')
'''

'''
part = 0.5

move_sub_dirs('C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation',
              'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
              0.63)


'''

part = 8745/103745

create_precense(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/bad_leaf', part)
create_precense(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/fruit',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/fruit', part)
create_precense(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\leaf',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\leaf', part)
create_precense(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\other',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\other', part)
create_precense(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\stem',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\stem', part)
