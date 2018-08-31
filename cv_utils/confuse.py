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


pics, true_Y = make_inputs("C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation")
'''y_predictions = apply_classification(pics,
                                     model_path="../models/mobilenet/2round/2018_08_29_22_17_500_epochs_round_1_3_classes.model",
                                     fix_function=lambda x: fix_map[x])'''


y_predictions = apply_classification(pics, "../models/mobilenet/2round/2018_08_30_1_49_500_epochs_round_1_5_classes.model")

leafs_indexes = np.where(np.isin(y_predictions, [0, 2, 3]))[0]
leafs_crop = pics[leafs_indexes.tolist()]

# give path to the second round model
m2 = apply_classification(leafs_crop, "../models/mobilenet/2round/2018_08_30_0_41_500_epochs_round_2_3_classes.model")
fix_classes(y_predictions, m2, leafs_indexes)
calc_acc(true_Y, y_predictions)

mat = confusion_matrix(y_predictions, true_Y)
print(mat)
