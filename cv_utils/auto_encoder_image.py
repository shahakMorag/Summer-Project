from keras.models import load_model
import cv2
import numpy as np

image = cv2.imread("C:\Tomato_Classification_Project\Tomato_Classification_Project\cropped_data\cropped/110432.png", 1).reshape(1, 372, 372, 3)

model = load_model("../models/encoder_decoder/autoencoder_2018_09_19_14_33.model")
res = model.predict(image)[0].astype(dtype=np.uint8)
dif = abs(image[0] - res)
cv2.imshow("orig", image[0])
cv2.imshow("res", res)
cv2.imshow("diff", dif)
cv2.waitKey(15000)

print("")
