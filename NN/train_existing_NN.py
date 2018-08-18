from keras.models import load_model

# the data, split between train and test sets
from NN.makeInputs import make_inputs

batch_size = 250
num_classes = 5
epochs = 100

limit = 5000
test = 100

img_rows, img_cols = 128, 128
channels = 3

x_train, y_train, x_test, y_test = make_inputs(8000, limit, test, num_classes, True)

input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = load_model("../NN/first.model")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("fourth.model")
