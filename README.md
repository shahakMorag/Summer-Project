# Semantic Segmentation Of Tomatoes

![tomato](https://dictionary.cambridge.org/images/thumb/tomato_noun_001_17860.jpg?version=4.0.30)

## first day
Today we learned how to use `PIL` library in Python in order to edit and work with images.

We covered various type of transformations of photos, colors and shapes.

![sharpened tomato](test\image transformations\Edge_Enhance_Tomato.jpg)


An example of sharpening photo in order to facilitate the learning the tomato. 
We used this line of code: 
```python
from PIL import Image, ImageFilter

image = Image.open("Tomato.jpg")
image.filter(ImageFilter.EDGE_ENHANCE_MORE)
```

We also tried invert photos: 

![inverted tomato](test\image transformations\inverted_Tomato.jpg)

with this code:
```python
from PIL import Image, ImageChops

image = Image.open("Tomato.jpg")
ImageChops.invert(image)
```

And we done a lot of other transformations.

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

