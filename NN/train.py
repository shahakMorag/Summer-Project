from NN.model import custom_network, changed_model
from NN.makeInputs import make_inputs

X, Y = make_inputs()

model = changed_model()  # custom_network()

model.fit(X, Y,
          n_epoch=5,
          show_metric=True,
          shuffle=True,
          run_id='tomato')

model.save('second.model')
