# Semantic Segmentation Of Tomatoes

![tomato](https://dictionary.cambridge.org/images/thumb/tomato_noun_001_17860.jpg?version=4.0.30)

## First Day

### Intro
Today we learned how to use `PIL` library in Python in order to edit and work with images.

We covered various type of transformations of photos, colors and shapes.

### PIL Features
![sharpened tomato](test/image%20transformations/Edge_Enhance_Tomato.jpg)


An example of sharpening photo in order to facilitate the learning the tomato. 
We used this line of code: 
```python
from PIL import Image, ImageFilter

image = Image.open("Tomato.jpg")
image.filter(ImageFilter.EDGE_ENHANCE_MORE)
```

We also tried invert photos: 

![inverted tomato](test/image%20transformations/Inverted_Tomato.jpg)

with this code:
```python
from PIL import Image, ImageChops

image = Image.open("Tomato.jpg")
ImageChops.invert(image)
```

And we done a lot of other transformations.

### Cropping
We have created a function that divides image to overlapping sub-images.
This is the code we used:
```python
def createCrops(im, step_x, step_y, crop_x, crop_y):
    res = []
    for up in range(0, im.size[1] - crop_y, step_y):
        for left in range(0, im.size[0] - crop_x, step_x):
            cropped = im.crop((left, up, left+crop_x, up+crop_y))
            res.append(cropped)

    return res
```

Here is part of the results for using crop on a 500x500 image. 

the parameters used are crop width and height of 150 and stride of 120:


![crop0](test/PIL%20tests/Crop0.jpg)
![crop1](test/PIL%20tests/Crop1.jpg)

![crop2](test/PIL%20tests/Crop2.jpg)
![crop3](test/PIL%20tests/Crop3.jpg)

### Color Spaces And Random Ones
We have experimented different types of color spaces transformations.

We wrote code that transform color space:
```python
# func is the function that changes the RGB triplet
# into other color space
def change_color_space(image, func):
    h, w = image.size
    res = Image.new("RGB", image.size, (255, 255, 255))

    for i in range(h):
        for j in range(w):
            res.putpixel((i, j), func(image.getpixel((i, j))))

    return res
```

First thing that we tried is to transform RGB into YUV with this function:
```python
import math
import numpy as np

def float_to_short(f):
    return math.ceil(f) % 256

def rgb_transform(A, rgb):
    r, g, b = rgb
    c = A @ np.array([r, g, b])
    r, g, b = float_to_short(c[0]), float_to_short(c[1]), float_to_short(c[2])
    return r, g, b

def rgb2uyv(rgb):
    A = np.array([[-0.1471, -0.2889, 0.4360],
                  [0.2990, 0.5870, 0.1140],
                  [0.6150, -0.5150, -0.1000]])

    return rgb_transform(A, rgb)
```
Here is photo of the result:

![YUV Tomato](test/image%20transformations/tomato_in_yuv_colorspace.jpg)

 Later we came up with the idea of make a random color space transformation.
 We hope that using it will improve the results of the learning.  We tried several distributions.
 Here is the code with exponential distribution:
 
 ```python
def random_color_space(image):
    image = image.convert("RGB")
    A = np.random.exponential(1/7, (3, 3))
    b = np.random.rand(3)
    return change_color_space(image, lambda x: add_tuples(rgb_transform(A, x), b))
```
 
Here is picture before transformation:

![eyal](test/image%20transformations/eyal.png)

Here is the picture after transformation ( the tomatoes gets different colors from the rest of the picture):

![eyal after transformation](test/image%20transformations/eyal%20transformation.png)

> We should be careful with using this method because wild distributions and
> parameters can lead to wild images for instance:

![wild tomato](test/image%20transformations/wild_tomato.jpeg) 
![wild tomato 2](test/image%20transformations/wild_tomato2.jpeg) 


  