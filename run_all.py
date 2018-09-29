from subprocess import Popen
import re
from os import path
import subprocess
import os
import logging
import time
import json

if __name__ == '__main__':
    # size 128x128

    with open('trainConfigs.json') as fp:
        args = json.load(fp)
    log_dir = path.join(args["logsDir"], args["modelName"])
    if not path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = path.join(*[args["logsDir"], args["modelName"], "log.log"])
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logging.info('Starting training process')
    start_time = time.time()

    # mobilenet training
    logging.info('Training Mobilenet')
    print('Training Mobilenet')
    output = Popen(['python', 'NN\keras_neural_net.py',
                    '-epochs=' + args['mobilenetTrainingEpochs'],
                    '-train_path=' + args['classifierTrainSetDir'],
                    '-val_path=' + args['classifierValidationSetDir'],
                    '-name=' + args["modelName"],
                    '-log_dir=' + log_dir], stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
    num_classes = int(re.search(r'num classes (\d+)', output).group(1))
    mobilenet_model_path = re.search(r'model path &(\S+)&', output).group(1)
    logging.info('Mobilenet trainined time: %d seconds' % (time.time() - start_time))
    print('Mobilenet trainined time: %d seconds' % (time.time() - start_time))
    logging.info('There are %d classes' % num_classes)
    print('There are %d classes' % num_classes)

    # create crops
    if args["skipCropsCreation"]:
        logging.info("Configured to skip creation of crops")
        print("Configured to skip creation of crops")
    else:
        creating_crops_start_time = time.time()
        logging.info('Creating crops for the encoder decoder')
        print('Creating crops for the encoder decoder')
        output = Popen(['python', 'cv_utils\create_patches_enc_dec.py', '-images_path=' + args['originalImagePath'], '-save_path=' + args['cropsSavePath']], stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
        logging.info('Crops created. Took %d seconds' % (time.time() - creating_crops_start_time))
        print('Crops created. Took %d seconds' % (time.time() - creating_crops_start_time))

    path_500 = path.join(args['cropsSavePath'], 'crops_500')
    path_372 = path.join(args['cropsSavePath'], 'crops_372')

    num_of_enc_dec_patches = len([file for file in os.listdir(path_500)])
    # create mobile net ground truth
    ground_truth_start_time = time.time()
    logging.info('Creating encoder decoder ground truth from the crops')
    print('Creating encoder decoder ground truth')
    jump = int(args["mobilenetSegmentationJump"])
    ground_truth_limit = min(int(args["limitGroundTruthCreation"]), num_of_enc_dec_patches)
    rest = ground_truth_limit % jump
    for start in range(0, ground_truth_limit - jump, jump):
        print('Created %d out of %d' % (start, ground_truth_limit))
        output = Popen(['python', 'cv_utils/run_on_several_images.py',
                        '-jump=' + str(jump),
                        '-start_image_number=' + str(start),
                        '-num_classes=' + str(num_classes),
                        '-model_path=' + mobilenet_model_path,
                        '-patches_path=' + path_500,
                        '-save_dir=' + args["saveMobilenetGroundTruthDir"]],
                       stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
    if rest:
        output = output = Popen(['python', 'cv_utils/run_on_several_images.py',
                                 '-jump=' + str(rest),
                                 '-start_image_number=' + str(start),
                                 '-num_classes=' + str(num_classes),
                                 '-model_path=' + mobilenet_model_path,
                                 '-patches_path=' + path_500,
                                 '-save_dir=' + args["saveMobilenetGroundTruthDir"]],
                                stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
    logging.info('Ground truth created in %d seconds' % (time.time() - ground_truth_start_time))
    print('Ground truth created in %d seconds' % (time.time() - ground_truth_start_time))


    # Training encoder decoder
    enc_dec_start_time = time.time()
    logging.info('Training encoder decoder!')
    print('Training encoder decoder!')
    # train our encoder decoder
    Popen(['python', 'enc_dec/train.py',
           '-classifier=' + args["modelName"],
           '-crops_dir=' + path_372,
           '-ground_truth_dir=' + args["saveMobilenetGroundTruthDir"],
           '-auto_encoder_n_max=' + str(ground_truth_limit),
           '-encoder_decoder_n_max=' + str(ground_truth_limit),
           '-auto_encoder_training_epochs=' + args["autoEncoderTrainingEpochs"],
           '-encoder_decoder_training_epochs=' + args["encoderDecoderTrainingEpochs"]], stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')

    logging.info('Encoder decoder training time is %d seconds' % (time.time() - enc_dec_start_time))
    print('Encoder decoder training time is %d seconds' % (time.time() - enc_dec_start_time))

    logging.info('Done! Total time is: %d seconds' % (time.time() - start_time))
    print('Done! Total time is: %d seconds' % (time.time() - start_time))
