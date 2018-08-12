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

> We should be careful when using this method because wild <br />
 distributions and parameters could lead to wild images, for instance:

![wild tomato](test/image%20transformations/wild_tomato.jpeg) 
![wild tomato 2](test/image%20transformations/wild_tomato2.jpeg) 


## Second Day

### Transformations
We will start working on Affine transformations to pictures.
First we the image to the center of all image that it won't cut after transformation.
For example this image:

![eyal cutted](test/image%20transformations/eyal_cuted.jpg) 

here is the picture after the change:

![middle eyal](test/image%20transformations/eyal_moved.jpg)

We created 3 different affine transformations on images in order to make more diversed data set for the training.

Here are some examples:

![first affine transformation](test/image%20transformations/Tomato_first_affine.jpg) 
![second affine transformation](test/image%20transformations/Tomato_second_affine.jpg) 
![third affine transformation](test/image%20transformations/Tomato_third_affine.jpg) 

We came up with the idea of refining the transformation from yesterday by making several transformation for every category.

### Neural Network
Finally we started building the neural network.

![neural network](https://icdn5.digitaltrends.com/image/artificial_neural_network_1-791x388.jpg)

#### Architecture
We created a neural network with the following architecture:
```python
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

NN = input_data(shape=[None, 128, 128, 3],
                # data_preprocessing=img_prep,
                name='input')

NN = conv_2d(NN, 32, 2, activation='relu')
NN = max_pool_2d(NN, 2)
NN = batch_normalization(NN)

NN = fully_connected(NN, 128, activation='relu', weights_init='xavier', bias_init='xavier')
NN = dropout(NN, 0.5)

NN = fully_connected(NN, 5, activation='softmax')

NN = regression(NN, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
``` 

We can see that we have shallow neural network with only one convolution part and one hidden layer.

This architecture is because our laptop's cpu is slower then the gpu's in the lab and we have only this for now.



### Preprocessing
We came up with the idea of generating data sets using external softwares, e.g. Matlab, since we will do it only once.

