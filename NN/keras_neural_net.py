from __future__ import print_function
from os import path
import time
import datetime
from mobilenet import get_model
from nn_utils import get_train_generator, get_valid_generator, get_callbacks
import argparse


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


def train(model_name, train_images_path, valid_images_path, input_shape, target_size, log_dir,
          save_dir="C:/Users\eitan.k\PycharmProjects\Summer-Project\models\smaller_mobilenet",
          reserve_layers=10, epochs=500, preprocessing_function=None, log=None):
    start_time = time.time()
    model_path = path.join(save_dir, get_start_date() + "_" + repr(epochs) + "_epochs_" + model_name + ".model")
    print('model path', '&' + model_path + '&')

    train_generator = get_train_generator(train_images_path, batch_size=250, target_size=target_size,
                                          preprocessing_function=preprocessing_function)
    num_classes = len(train_generator.class_indices)
    print('#class indices ' + str(train_generator.class_indices) + '#')
    print('num classes', num_classes)
    valid_generator = get_valid_generator(valid_images_path, target_size, preprocessing_function=preprocessing_function)

    model = get_model(input_shape, num_classes=num_classes, reserve_layers=reserve_layers)

    # print('Train indices: ' + str(train_generator.class_indices))
    # print('Validate indices: ' + str(valid_generator.class_indices))

    callbacks = get_callbacks(validate_freq=10,
                              valid_generator=valid_generator,
                              model=model,
                              patience=5,
                              save_freq=10,
                              model_path=model_path,
                              log_dir=log_dir)

    model.fit_generator(generator=train_generator,
                        epochs=epochs,
                        verbose=1,
                        workers=8,
                        callbacks=callbacks)

    training_time = time.time() - start_time
    # print("Total training time is:", training_time)
    model.save(model_path)

    # print("Evaluating...")
    score_seg = model.evaluate_generator(
        generator=valid_generator,
        verbose=1,
        workers=8,
    )
    print('#val_acc: %f#' % (score_seg[1],))

    if log is not None:
        log.write(model_name + ':\nTraining time - %d seconds\nValidate Accuracy - %s\n\n\n' % (
            training_time, str(score_seg[1])))
        log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True)
    parser.add_argument('-train_path', required=True)
    parser.add_argument('-val_path', required=True)
    parser.add_argument('-epochs', required=True, type=int)
    parser.add_argument('-log_dir', required=True)
    parser.add_argument('-patch_size', type=int) #[shahak] need to check
    parser.add_argument('-save_dir')
    args = parser.parse_args()

    patch_size = args.patch_size if args.patch_size is not None else 128
    train(model_name=args.name,
          train_images_path=args.train_path,
          valid_images_path=args.val_path,
          input_shape=(patch_size, patch_size, 3),
          target_size=(patch_size, patch_size), #[shahak] check what target size is
          epochs=args.epochs,
          log_dir=args.log_dir,
          save_dir=args.save_dir)
