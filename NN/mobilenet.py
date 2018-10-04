import keras
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten, MaxPool2D, Conv2D, Activation, ZeroPadding2D
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import adadelta
from keras.models import Model
from keras.utils import vis_utils
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin'

def get_model(input_shape, num_classes, reserve_layers=15):
    mobile = keras.applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=input_shape, dropout=0.25)

    x = mobile.layers[reserve_layers].output
    x = MaxPool2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=mobile.input, outputs=predictions)

    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    # print('There are', len(model.layers), 'layers')
    return model


if __name__ == '__main__':
    model = get_model(input_shape=(128, 128, 3), num_classes=5, reserve_layers=10)
    vis_utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="classifier_plot.png")