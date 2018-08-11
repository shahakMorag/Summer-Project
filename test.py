from PIL import Image
im = Image.open("test.png")
#rotate image
im.rotate(10).show()
