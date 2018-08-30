import keras
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten, MaxPool2D
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import adadelta
from keras.models import Model


def get_model(input_shape, num_classes, reserve_layers=11):
    mobile = keras.applications.mobilenet.MobileNet(input_shape=input_shape, dropout=0.25)

    x = mobile.layers[reserve_layers].output
    x = MaxPool2D(pool_size=(15, 15))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=mobile.input, outputs=predictions)

    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    print('There are', len(model.layers), 'layers')
    return model


#m = get_model((128, 128, 3), 5)
