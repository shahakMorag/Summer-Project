from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# the data, split between train and test sets
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

batch_size = 200
epochs = 3
patience = 30
path_to_data = 'D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5'
path_to_load_model = "../NN/first.model"
path_to_save_model = "second.model"
seed = 1


def train(model):
    # callbacks
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    callbacks = [early_stop, reduce_lr]

    print('Starting to fit the model...')

    train_data_gen = ImageDataGenerator(
        validation_split=0.0125,
        rescale=1. / 255,
    )

    train_generator = train_data_gen.flow_from_directory(
        path_to_data,
        target_size=(128, 128),
        batch_size=batch_size,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical',
        seed=seed
    )

    print(train_generator.class_indices)

    model.fit_generator(
        train_generator,
        epochs=epochs,
        verbose=1,
        workers=16,
        callbacks=callbacks)


model = load_model(path_to_load_model)
train(model)
model.save(path_to_save_model)
