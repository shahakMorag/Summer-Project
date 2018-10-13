# How to use our code
## In order to run the code you need the following libraries
* tensorflow
* keras
* opencv-python
* numpy

## Train mobilenet classifier
You have to run the following:
```
python -name="" -train_path="" -val_path="" -epochs="" -log_dir="" 
```
### Parameters explanation
* __name__ the model will be save in models/smaller_mobilenet/{start training date}\_{number of epochs}\_{name}.model
* __train\_path__ path to folder which contains folder of images for each class the the model will be trained on
* __val\_path__ path to folder which contains directory for each class the the model will be trained on and used for validation
* __epochs__ number of epochs to train the mobilenet
* __log\_dir__ where to save the logs from the training
* __patch\_size__ the size of the patches to train the mobile net has default of 128

## Train the whole Encoder-Decoder

You have to run the following:

```
python enc_dec\train.py -classifier="" -ground_truth_dir="" 
    -crops_dir="" -auto_encoder_n_max="" -encoder_decoder_n_max=""
```

### Parameters explanation

* __classifier__ the name of the model that we will save to models/encoder_decoder/ dir
* __ground_truth_dir__ dir with the results of running the mobilenet
* __crops_dir__ dir with the crops to train the auto encoder and the mobile net
* __auto_encoder_n_max__ number of images to train the auto encoder
* __encoder_decoder_n_max__ number of images to train the Encoder-Decoder

## In order to run all the training process
 You have to run the following:
 ```
python run_all.py
```

If you want to config parameters for the training then change trainConfig.json parameters.
