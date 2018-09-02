import numpy as np
from makeInputs import make_inputs
from sklearn.metrics import confusion_matrix

from cv_utils.crop_utils import apply_classification, fix_classes

fix_map = dict(zip([0, 1, 2], [1, 3, 4]))


def calc_acc(truth, predictions):
    if len(truth) != len(predictions):
        print("Error! Arrays lengths don't match!")
        return

    acc = np.sum(np.array(truth) == np.array(predictions)) / len(truth)
    print("acc:", acc)


pics, true_Y = make_inputs("C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation_128")
y_predictions = apply_classification(pics,
                                     model_path="../models/mobilenet/all_models/2018_09_01_13_13_500_epochs_rgb_128_round_1.model",
                                     fix_function=lambda x: fix_map[x])


# y_predictions = apply_classification(pics, "../models/mobilenet/all_models/2018_09_01_13_13_500_epochs_rgb_128_round_1.model")

leafs_indexes = np.where(np.isin(y_predictions, [3]))[0]
leafs_crop = pics[leafs_indexes.tolist()]

# give path to the second round model
m2 = apply_classification(leafs_crop, "../models/mobilenet/all_models/2018_09_01_14_59_500_epochs_rgb_128_round_2.model")
fix_classes(y_predictions, m2, leafs_indexes)
calc_acc(true_Y, y_predictions)

mat = confusion_matrix(y_predictions, true_Y)
print(mat)
