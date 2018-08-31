from sklearn.metrics import confusion_matrix
import time
import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
from makeInputs import make_inputs


def apply_classification(image_list, model_path, batch_size=1):
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


fix_map = dict(zip([0, 1, 2], [1, 3, 4]))
maps2 = dict(zip([0, 1, 2], [0, 2, 3]))


def fix_classes(m, m2, leafs_indexes):
    i = 0
    while i < len(m2):
        # the 2 is because the numbers are only in [0,1]
        m[leafs_indexes[i]] = maps2.__getitem__(m2[i])
        i += 1


def calc_acc(truth, predictions):
    if len(truth) != len(predictions):
        print("Error! Arrays lengths don't match!")
        return

    total = len(truth)
    success = 0
    for i in range(total):
        if truth[i] == predictions[i]:
            success += 1

    acc = success / total
    print("acc:", acc)


pics, true_Y = make_inputs("C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation")
'''Y_pred = apply_classification(pics, model_path="../models/mobilenet/2round/2018_08_29_22_17_500_epochs_round_1_3_classes.model")
for i in range(len(true_Y)):
    print('[',true_Y[i],",",Y_pred[i],"]")
for i in range(len(Y_pred)):
    Y_pred[i] = fix_map.__getitem__(Y_pred[i])'''

Y_pred = apply_classification(pics,
                              model_path="../models/mobilenet/2round/2018_08_30_1_49_500_epochs_round_1_5_classes.model")

'''print("Accuracy 1-round:")
calc_acc(true_Y, Y_pred)

mat = confusion_matrix(Y_pred, true_Y)
print(mat)'''

''' ------------------------- second round ------------------------- '''

leafs_indexes = np.where(np.isin(Y_pred, [0, 2, 3]))[0]
leafs_crop = pics[leafs_indexes.tolist()]

# give path to the second round model
m2 = apply_classification(leafs_crop,
                          model_path="../models/mobilenet/2round/2018_08_30_0_41_500_epochs_round_2_3_classes.model")
fix_classes(Y_pred, m2, leafs_indexes)
print("Accuracy 1-round:")
calc_acc(true_Y, Y_pred)

mat = confusion_matrix(Y_pred, true_Y)
print(mat)
