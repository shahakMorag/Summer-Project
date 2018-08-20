import os
import shutil
import random

def create(source_path, dest_path, part):
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


part = 0.05

create(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/bad_leaf', part)
create(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/fruit',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation/fruit', part)
create(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\leaf',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\leaf', part)
create(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\other',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\other', part)
create(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5\stem',
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation\stem', part)
