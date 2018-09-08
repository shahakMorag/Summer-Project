import keras
from keras.layers import MaxPool2D, Conv2D, DepthwiseConv2D
from keras.optimizers import Adam
from keras.models import Model


def encoder_decoder(input_shape, reserve_layers=30):
    mobile = keras.applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=input_shape, dropout=0.25)

    x = mobile.layers[reserve_layers].output
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = DepthwiseConv2D((8, 8), activation="relu")(x)
    x = Conv2D(5, (1, 1))(x)

    model = Model(inputs=mobile.input, outputs=x)

    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


