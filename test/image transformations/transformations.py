from PIL import Image, ImageChops, ImageFilter

im = Image.open("Tomato.jpg")
im2 = im.convert("HSV")
print(im2.mode)