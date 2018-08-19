from keras.models import load_model

# the data, split between train and test sets
from NN.makeInputs import make_inputs
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

batch_size = 200
num_classes = 5
epochs = 150

def train(start, limit, model, log_file_path,trained_models_path, patience = 30):
    test = 100

    img_rows, img_cols = 128, 128
    channels = 3

    x_train, y_train, x_test, y_test = make_inputs(start, limit, test, num_classes, True)

    input_shape = (img_rows, img_cols, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #callbacks
    #csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)

    callbacks = [model_checkpoint, early_stop, reduce_lr]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks= callbacks)
    #callbacks
    #csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)

    callbacks = [model_checkpoint, early_stop, reduce_lr]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              callbacks= callbacks,
              validation_split=0.0125)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



model = load_model("../NN/first.model")
limit = 8000
train(0, limit, model)
train(8000, limit, model)
train(16000, limit, model)
train(24000, limit, model)

model.save("second.model")
