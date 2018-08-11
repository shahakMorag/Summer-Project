from PIL import Image, ImageFilter
import numpy as np
import random

im1 = Image.open("test.png")
im2 = Image.open("test2.jpg")
im3 = Image.open("red.png")

size = 500, 500

background = Image.new("RGBA", im1.size, (255, 255, 255))
background.paste(im1)
im1 = background

background = Image.new("RGBA", im2.size, (255, 255, 255))
background.paste(im2)
im2 = background

im1 = im1.resize(size, Image.ANTIALIAS)
im2 = im2.resize(size, Image.ANTIALIAS)

#Image.alpha_composite(im1, im2).show()

#Image.blend(im1, im2, 0.3).show()

#Image.eval(im2, lambda x: 3 * x % 256).show()

#im1.convert(mode="L").show()

#im1.convert(mode="1").show()

#Image.fromarray(np.random.randn(1000,1000), mode='RGBA').show()

#im2.crop((20, 20, 80, 80)).show()

#im1.filter(ImageFilter.CONTOUR).show()

#im1.filter(ImageFilter.FIND_EDGES).show()

#im1.filter(ImageFilter.EDGE_ENHANCE_MORE).show() #important

#print(im1.getbbox()) #output (0, 0, 500, 500)

#print(im1.histogram())

'''r, g, b, a = im1.split()

blanck = Image.fromarray(np.zeros(size), mode="L")

Image.merge('RGBA', (r, blanck, blanck, a)).show()
Image.merge('RGBA', (blanck, g, blanck, a)).show()
Image.merge('RGBA', (blanck, blanck, b, a)).show()
'''


