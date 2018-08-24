import keras
from keras import Sequential, Model, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import inception_v3


def get_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    base_model1 = keras.applications.densenet.DenseNet201(input_tensor=input_tensor, weights='imagenet', include_top=False)
    base_model2 = keras.applications.nasnet.NASNetMobile(input_tensor=input_tensor, weights='imagenet', include_top=False)

    x = keras.layers.concatenate([base_model1.output, base_model2.output])
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=keras.layers.concatenate([base_model1.input, base_model2.input]), outputs=predictions)

    for layer in base_model1.layers:
        layer.trainable = False

    for layer in base_model2.layers:
        layer.trainable = False

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adadelta(lr=4),
                  metrics=['accuracy'])

    return model
