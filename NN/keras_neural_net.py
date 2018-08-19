from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, K, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from makeInputs import make_inputs

batch_size = 64
num_classes = 5
epochs = 1
limit = 8000
test = 100

# input image dimensions
img_rows, img_cols = 128, 128
channels = 3

# the data, split between train and test sets
x_train, y_train = make_inputs(0, limit, num_classes)

input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_train /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

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

trained_models_path = 'C:/Users\eitan.k\PycharmProjects\Summer-Project\models/'
patience = 30

# callbacks
early_stop = EarlyStopping('acc', patience=patience)
reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                              patience=int(patience / 4), verbose=1)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'acc', verbose=1, save_best_only=True, period=40)

callbacks = [model_checkpoint, early_stop, reduce_lr]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_split=0.0125)


model.save("first.model")
