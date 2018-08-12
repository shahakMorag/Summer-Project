from PIL import Image
import glob

def make_inputs(dir):
    images = []
    limit = 100
    for filename in glob.glob(dir + '/*.png'):
        limit -= 1
        if limit == 0:
            break
        images.append(Image.open(filename))
    return images

make_inputs("F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/bad_leaf")
make_inputs("F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/fruit")
make_inputs("F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/leaf")
make_inputs("F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/other")
make_inputs("F:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5/stem")