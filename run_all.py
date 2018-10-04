import datetime
from subprocess import Popen
import re
from os import path
import subprocess
import os
import logging
import time
import json


def get_start_date():
    return str(datetime.date.today()).replace('-', '_') + "_" + str(datetime.datetime.now().hour) + "_" + str(
        datetime.datetime.now().minute)


if __name__ == '__main__':
    # size 128x128
    with open('trainConfigs.json') as fp:
        args = json.load(fp)
    log_dir = path.join(args["logsDir"], args["modelName"])
    if not path.exists(log_dir):
        os.mkdir(log_dir)

    log_base_path = path.join(*[args["logsDir"], args["modelName"], get_start_date()])
    log_file = path.join(log_base_path, "run_all.log")
    mobilenet_log_file = path.join(log_base_path, "mobilenet.log")
    enc_dec_log_file = path.join(log_base_path, "enc_dec.log")
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    logging.info('Starting training process')
    start_time = time.time()

    # mobilenet training
    if args["skipMobilenNetTraining"]:
        logging.info("Configured to skip mobilenet training")
        print("Configured to skip mobilenet training")
        mobilenet_model_path = args["mobilenet_path"]
        num_classes = int(args['num_classes'])
    else:
        logging.info('Training Mobilenet')
        print('Training Mobilenet')
        output = Popen(['python', 'NN\keras_neural_net.py',
                        '-epochs=' + args['mobilenetTrainingEpochs'],
                        '-train_path=' + args['classifierTrainSetDir'],
                        '-val_path=' + args['classifierValidationSetDir'],
                        '-name=' + args["modelName"],
                        '-log_dir=' + log_dir], stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
        with open(mobilenet_log_file, 'w') as f:
            f.write(output)

        num_classes = int(re.search(r'num classes (\d+)', output).group(1))
        mobilenet_model_path = re.search(r'model path &(\S+)&', output).group(1)
        mobilenet_acc = re.search(r'#val_acc: (\d+\.\d+)#', output).group(1)
        class_indices = re.search(r'#class indices (.+)#', output).group(1)
        logging.info('Mobilenet trainined time: %d seconds' % (time.time() - start_time))
        logging.info('Mobilenet val_acc: %s' % (mobilenet_acc,))
        logging.info('Class indices: %s' % (class_indices,))
        print('Mobilenet val_acc: %s' % (mobilenet_acc,))
        print('Mobilenet trainined time: %d seconds' % (time.time() - start_time))
        logging.info('There are %d classes' % num_classes)
        print('There are %d classes' % num_classes)

    # create crops
    crops_save_path = path.join(args['cropsSavePath'], args['modelName'])
    if not path.exists(crops_save_path): os.mkdir(crops_save_path)
    if args["skipCropsCreation"]:
        logging.info("Configured to skip creation of crops")
        print("Configured to skip creation of crops")
    else:
        creating_crops_start_time = time.time()
        logging.info('Creating crops for the encoder decoder')
        print('Creating crops for the encoder decoder')
        output = Popen(['python', 'cv_utils\create_patches_enc_dec.py',
                        '-images_path=' + args['originalImagePath'],
                        '-save_path=' + crops_save_path],
                       stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
        logging.info('Crops created. Took %d seconds' % (time.time() - creating_crops_start_time))
        print('Crops created. Took %d seconds' % (time.time() - creating_crops_start_time))

    path_500 = path.join(crops_save_path, 'crops_500')
    path_372 = path.join(crops_save_path, 'crops_372')

    num_of_enc_dec_patches = len([file for file in os.listdir(path_500)])
    print("There are %d crops" % (num_of_enc_dec_patches,))
    logging.info("There are %d crops" % (num_of_enc_dec_patches,))

    # create mobile net ground truth
    ground_truth_start_time = time.time()
    logging.info('Creating encoder decoder ground truth from the crops')
    print('Creating encoder decoder ground truth')
    jump = int(args["mobilenetSegmentationJump"])
    ground_truth_limit = min(int(args["limitGroundTruthCreation"]), num_of_enc_dec_patches)
    rest = ground_truth_limit % jump
    ground_truth_save_path = path.join(args["saveMobilenetGroundTruthDir"], args['modelName'])
    if not path.exists(ground_truth_save_path):
        os.mkdir(ground_truth_save_path)
    for start in range(0, ground_truth_limit - jump, jump):
        print('Created %d out of %d' % (start, ground_truth_limit))
        output = Popen(['python', 'cv_utils/run_on_several_images.py',
                        '-jump=' + str(jump),
                        '-start_image_number=' + str(start),
                        '-num_classes=' + str(num_classes),
                        '-model_path=' + mobilenet_model_path,
                        '-patches_path=' + path_500,
                        '-save_dir=' + ground_truth_save_path],
                       stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
    if rest:
        print('Created %d out of %d' % (start+jump, ground_truth_limit))
        output = Popen(['python', 'cv_utils/run_on_several_images.py',
                        '-jump=' + str(rest),
                        '-start_image_number=' + str(start+jump),
                        '-num_classes=' + str(num_classes),
                        '-model_path=' + mobilenet_model_path,
                        '-patches_path=' + path_500,
                        '-save_dir=' + ground_truth_save_path],
                       stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')
    logging.info('Ground truth created in %d seconds' % (time.time() - ground_truth_start_time))
    print('Ground truth created in %d seconds' % (time.time() - ground_truth_start_time))

    # Training encoder decoder
    enc_dec_start_time = time.time()
    logging.info('Training encoder decoder!')
    print('Training encoder decoder!')
    # train our encoder decoder
    output = Popen(['python', 'enc_dec/train.py',
           '-classifier=' + args["modelName"],
           '-crops_dir=' + path_372,
           '-ground_truth_dir=' + ground_truth_save_path,
           '-auto_encoder_n_max=' + str(num_of_enc_dec_patches),
           '-encoder_decoder_n_max=' + str(ground_truth_limit)],
            stdout=subprocess.PIPE).stdout.read().decode(encoding='utf-8')

    with open(enc_dec_log_file, 'w') as f:
        f.write(output)


    logging.info('Encoder decoder training time is %d seconds' % (time.time() - enc_dec_start_time))
    print('Encoder decoder training time is %d seconds' % (time.time() - enc_dec_start_time))

    logging.info('Done! Total time is: %d seconds' % (time.time() - start_time))
    print('Done! Total time is: %d seconds' % (time.time() - start_time))
