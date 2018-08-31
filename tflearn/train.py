from tflearn.model import changed_model
from cv_utils.makeInputs import make_inputs

X, Y = make_inputs()

model = changed_model()  # custom_network()

model.fit(X, Y,
          n_epoch=5,
          show_metric=True,
          shuffle=True,
          run_id='tomato')

model.save('second.model')
