from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# the data, split between train and test sets
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from nn_utils import get_train_generator, get_valid_generator

batch_size = 200
epochs = 500
patience = 20
train_images_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5'
valid_images_path = 'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches/validation'
path_to_load_model = "../models/2008182.model"
path_to_save_model = "../models/200818_40_epochs.model"
seed = 1


def train(model):
    # callbacks
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=1, verbose=1)
    callbacks = [early_stop, reduce_lr]

    print('Starting to fit the model...')

    # ------------------------------------------ training data set ------------------------------------------

    train_generator = get_train_generator(train_images_path, batch_size)
    print(train_generator.class_indices)

    # ------------------------------------------ validation data set ----------------------------------------

    valid_generator = get_valid_generator(valid_images_path)

    # ------------------------------------------ callbacks --------------------------------------------------

    # callbacks
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)

    callbacks = [early_stop, reduce_lr]

    model.fit_generator(
        generator=train_generator,
        #validation_data=valid_generator,
        epochs=epochs,
        verbose=1,
        workers=8,
        callbacks=callbacks)


model = load_model(path_to_load_model)
train(model)
model.save(path_to_save_model)

print("Finished training!")