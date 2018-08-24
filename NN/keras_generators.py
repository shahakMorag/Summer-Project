from keras_preprocessing.image import ImageDataGenerator


def get_train_generator(train_images_path, batch_size):
    seed = 1

    train_data_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_data_gen.flow_from_directory(
        train_images_path,
        target_size=(128, 128),
        batch_size=batch_size,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical',
        seed=seed,
        shuffle=True
    )

    return train_generator


def get_valid_generator(valid_images_path):
    valid_data_gen = ImageDataGenerator(
        rescale=1. / 255,
    )

    valid_generator = valid_data_gen.flow_from_directory(
        valid_images_path,
        target_size=(128, 128),
        batch_size=1,
        color_mode='rgb',
        classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
        class_mode='categorical',
        shuffle=False
    )

    return valid_generator