import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, InputLayer, ZeroPadding2D

import keras.backend as K
def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5,5),
                     activation='relu',
                     input_shape=input_shape,
                     ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=lambda y, f: tilted_loss(0.5, y, f),
                  optimizer=keras.optimizers.Adam(lr=10),
                  metrics=['accuracy'])

    model.summary()
    print('There are', len(model.layers), 'layers')

    return model


m = get_model((128, 128, 3), 5)