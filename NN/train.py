from NN.model import custom_network
from NN.makeInputs import make_inputs

X, Y = make_inputs()

model = custom_network()

model.fit(X, Y,
          n_epoch=100,
          show_metric=True,
          shuffle=True,
          run_id='tomato')

model.save('first.model')
