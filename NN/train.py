from NN.model import custom_network
from NN.makeInputs import make_inputs

X, Y = make_inputs()

model = custom_network()

model.fit({'input': X}, {'targets': Y},
          n_epoch=3,
          show_metric=True,
          run_id='tomato')
