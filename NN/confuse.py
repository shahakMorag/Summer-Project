from sklearn.metrics import confusion_matrix
import time
import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from makeInputs import get_pictures


def apply_classification(image_list, batch_size=1, model_path='../models/mobilenet/2018_08_25_17_6_500_epochs.model'):
    start_time = time.time()
    print("Applying classification...")
    model = load_model(model_path)

    test_generator = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) \
        .flow(x=np.array(image_list),
              batch_size=batch_size,
              shuffle=False,
              )

    predicts = model.predict_generator(test_generator,
                                       steps=len(image_list),
                                       verbose=1,
                                       workers=16)

    tags = predicts.argmax(axis=1)
    end_time = time.time()
    d_time = end_time - start_time
    print("Classification took " + repr(d_time) + " seconds")
    return np.array(tags).flatten()


maps = dict(zip([0,1,2],[0,2,3]))


def fix_classes(m, m2, leafs_indexes):
    i = 0
    while i < len(m2):
        # the 2 is because the numbers are only in [0,1]
        m[leafs_indexes[i]] = maps.__getitem__(m2[i])
        i += 1


#valid_generator = get_valid_generator(valid_images_path="C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation")

pics, true_Y = get_pictures("C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation")
Y_pred = apply_classification(pics)

leafs_indexes = np.where(np.isin(Y_pred, [0, 2, 3]))[0]
leafs_crop = pics[leafs_indexes.tolist()]

# give path to the second round model
m2 = apply_classification(leafs_crop, model_path="../models/mobilenet/2018_08_26_20_47_500_epochs_leaf.model")
fix_classes(Y_pred, m2, leafs_indexes)

mat = confusion_matrix(Y_pred, true_Y)
print(mat)


