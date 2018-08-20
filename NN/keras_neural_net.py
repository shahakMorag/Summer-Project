from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

from NN.our_model import get_model

batch_size = 64
num_classes = 5
epochs = 1
limit = 2000
test = 100

# input image dimensions
img_rows, img_cols = 128, 128
channels = 3

# the data, split between train and test sets
x_train, y_train = make_inputs(0, limit, num_classes)

input_shape = (img_rows, img_cols, channels)
'''
x_train = x_train.astype('float32')
x_train /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = our_model.get_model(input_shape, num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 2),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2048, activation="tanh"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

patience = 30

# callbacks
early_stop = EarlyStopping('acc', patience=patience)
reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                              patience=int(patience / 4), verbose=1)

callbacks = [early_stop, reduce_lr]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_split=0.0125)


'''

patience = 30
model = get_model()

print('Starting to fit the model...')

train_datagen = ImageDataGenerator(
    validation_split=0.0125,
    rescale=1. / 255,
)

train_generator = train_datagen.flow_from_directory(
    'C:\Tomato_Classification_Project\Tomato_Classification_Project\Patches\Patches\patches_size_128_skip_32_categories_5',
    target_size=(128, 128),
    batch_size=batch_size,
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
    train_generator,
    epochs=epochs,
    verbose=1,
    workers=8,
    callbacks=callbacks)

model.save("20-8-18-1.model")
