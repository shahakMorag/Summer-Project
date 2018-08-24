import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense


def get_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(10, 10),
                     activation='relu',
                     input_shape=input_shape,
                     ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=10),
                  metrics=['accuracy'])

    return model
