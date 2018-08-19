from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# the data, split between train and test sets
from makeInputs import make_inputs
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

batch_size = 200
num_classes = 5
epochs = 3


def train(start, limit, model, trained_models_path, path_to_data, patience=30):
    '''
    x_train, y_train = make_inputs(start, limit, num_classes)

    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    #callbacks
    #csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    # model_checkpoint = ModelCheckpoint(model_names, 'acc', verbose=1, save_best_only=True, period=40)

    callbacks = [early_stop, reduce_lr]

    print('Starting to fit the model...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks= callbacks,
              validation_split=0.0125)
    '''
    # callbacks
    # csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    # model_checkpoint = ModelCheckpoint(model_names, 'acc', verbose=1, save_best_only=True, period=40)

    callbacks = [early_stop, reduce_lr]

    print('Starting to fit the model...')

    train_datagen = ImageDataGenerator(
        validation_split=0.0125,
        rescale=1. / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        path_to_data,
        target_size=(128, 128),
        batch_size=batch_size,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical'
    )

    model.fit_generator(
        train_generator,
        epochs=epochs,
        verbose=1,
        workers=16,
        callbacks=callbacks)


path_to_data = 'D:\Tomato_Classification_Project_5_iter\Patches\Patches\patches_size_128_skip_32_categories_5'

model = load_model("../NN/first.model")

train(0, 0, model, 'C:/Users\eitan.k\PycharmProjects\Summer-Project\models/', path_to_data)
model.save("second.model")
