from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# the data, split between train and test sets
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

batch_size = 200
epochs = 20
patience = 30
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

    train_data_gen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_data_gen.flow_from_directory(
        train_images_path,
        target_size=(128, 128),
        batch_size=batch_size,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical',
        seed=seed
    )

    # ------------------------------------------ validation data set ------------------------------------------
    valid_data_gen = ImageDataGenerator(
        rescale=1. / 255
    )

    valid_generator = valid_data_gen.flow_from_directory(
        valid_images_path,
        target_size=(128, 128),
        batch_size=1,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical'
    )

    print(train_generator.class_indices)

    # callbacks
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)

    callbacks = [early_stop, reduce_lr]

    model.fit_generator(
        generator=train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        verbose=1,
        workers=8,
        callbacks=callbacks)


model = load_model(path_to_load_model)
train(model)
model.save(path_to_save_model)

print("Finished training!")