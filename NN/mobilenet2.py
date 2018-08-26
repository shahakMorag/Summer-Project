import keras
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import adadelta
from keras.models import Model


def get_model(input_shape, num_classes):
    mobile = keras.applications.mobilenet.MobileNet(input_shape=input_shape, dropout=0.25)

    x = mobile.layers[-1].output
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=mobile.input, outputs=predictions)


    '''for layer in model.layers[:-85]:
        layer.trainable = False
    '''

    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model


m = get_model((128, 128, 3), 5)
