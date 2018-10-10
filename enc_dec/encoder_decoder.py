import os
import keras
from keras.layers import Input, Convolution2D, Lambda, UpSampling2D, AvgPool2D, GaussianNoise
from keras.models import Model, load_model
from keras.utils import vis_utils


def auto_encoder_avg_pooling(shape):
    input_img = Input(shape=shape)  # adapt this if using `channels_first` image data format
    kernels = 40
    x = Lambda(lambda x: x / 255.0)(input_img)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(input_img)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = AvgPool2D((2, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = GaussianNoise(0.05)(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)

    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same', name='encoder_output')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(kernels, (3, 3), activation='relu', padding="same")(x)
    x = Convolution2D(kernels, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, (3, 3), activation="sigmoid", padding='same')(x)
    decoded = Lambda(lambda x: x * 255.0)(decoded)

    auto_encoder = Model(input_img, decoded)
    auto_encoder.compile(optimizer="adadelta", loss=keras.losses.MSE, metrics=["accuracy"])
    return auto_encoder


def our_semantic_segmentation(auto_encoder_path , auto_encoder=None):
    if auto_encoder is None:
        auto_encoder = load_model(auto_encoder_path)
    for layer in auto_encoder.layers:
        layer.trainable = False

    x = auto_encoder.get_layer('encoder_output').output

    x = Convolution2D(30, (3, 3), activation="relu", padding="same", name='a1')(x)
    x = AvgPool2D((2, 2), padding="same", name='a2')(x)
    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a3')(x)
    x = AvgPool2D((2, 2), padding="same", name='a4')(x)
    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a5')(x)

    x = Convolution2D(32, (3, 3), activation="relu", padding="same", name='a6')(x)
    x = UpSampling2D((2, 2), name='a7')(x)
    x = Convolution2D(5, (3, 3), activation="relu", padding="same", name='a8')(x)

    model = Model(inputs=auto_encoder.input, outputs=x)
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
    return model


if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin'
    model = auto_encoder_avg_pooling((372, 372, 3))
    vis_utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="auto_encoder_plot.png")
    model = our_semantic_segmentation('stam', model)
    vis_utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file="enc_dec_plot.png")

