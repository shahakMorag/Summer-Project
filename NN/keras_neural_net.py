from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from inception_neural_net import get_model

# input image dimensions and parameters
input_shape = (128, 128, 3)
patience = 30
batch_size = 128
epochs = 8
seed = 1
train_images_path = 'C:\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5'

#
model = get_model(input_shape, num_classes=5)

print('Starting to fit the model...')

train_datagen = ImageDataGenerator(
    validation_split=0.0125,
    rescale=1. / 255,
)

train_generator = train_datagen.flow_from_directory(
    train_images_path,
    target_size=(128, 128),
    batch_size=batch_size,
    color_mode='rgb',
    classes=['bad_leaf', 'fruit', 'leaf', 'other', 'stem'],
    class_mode='categorical',
    seed=seed
)

print(train_generator.class_indices)

# callbacks
early_stop = EarlyStopping('acc', patience=patience)
reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                              patience=int(patience / 4), verbose=1)

callbacks = [early_stop, reduce_lr]

model.fit_generator(
    train_generator,
    epochs=epochs,
    verbose=1,
    workers=8,
    callbacks=callbacks)

model.save("2008181.model")
