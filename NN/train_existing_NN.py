from keras.models import load_model

# the data, split between train and test sets
from makeInputs import make_inputs
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

batch_size = 200
num_classes = 5
epochs = 50


def train(start, limit, model, trained_models_path, patience = 30):
    x_train, y_train = make_inputs(start, limit, num_classes)

    x_train = x_train.astype('float32')
    x_train /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    #callbacks
    #csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('acc', factor=0.1,
                                  patience=int(patience / 4), verbose=1)
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    # model_checkpoint = ModelCheckpoint(model_names, 'acc', verbose=1, save_best_only=True, period=40)

    callbacks = [early_stop, reduce_lr]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks= callbacks,
              validation_split=0.0125)

model = load_model("../NN/first.model")
limit = 20000

for i in range(25):
    train(i * limit, limit, model, 'C:/Users\eitan.k\PycharmProjects\Summer-Project\models/')

model.save("second.model")
