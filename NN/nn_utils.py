import keras
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator


def get_train_generator(train_images_path, batch_size, target_size):
    seed = 1

    train_data_gen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet.preprocess_input,
        # rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_data_gen.flow_from_directory(
        train_images_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        seed=seed,
        shuffle=True
    )

    return train_generator


def get_valid_generator(valid_images_path, target_size):
    valid_data_gen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet.preprocess_input,
        # rescale=1. / 255,
    )

    valid_generator = valid_data_gen.flow_from_directory(
        valid_images_path,
        target_size=target_size,
        batch_size=1,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )

    return valid_generator


class ValidateCallback(Callback):

    def __init__(self, p, valid_generator, model):
        super().__init__()
        self.param = p
        self.valid_generator = valid_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if not epoch % self.param == 0:
            return

        print("Evaluating...")
        scoreSeg = self.model.evaluate_generator(
            generator=self.valid_generator,
            verbose=1,
            workers=8,
        )

        print('Loss: ' + repr(scoreSeg[0]) + ', Acc: ' + repr(scoreSeg[1]))


class SaveCallback(Callback):

    def __init__(self, save_freq, model_path, model):
        super().__init__()
        self.model = model
        self.model_path = model_path
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if not epoch % self.save_freq == self.save_freq - 1:
            return

        self.model.save(self.model_path)

def get_callbacks(validate_freq, valid_generator, model, patience, save_freq, model_path):
    valid_callback = ValidateCallback(validate_freq, valid_generator, model)
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.6,
                                  patience=2, verbose=1)

    checkpoint = SaveCallback(save_freq, model_path, model)

    tensorboard = TensorBoard(log_dir='../logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None)

    return [valid_callback, early_stop, reduce_lr, tensorboard, checkpoint]