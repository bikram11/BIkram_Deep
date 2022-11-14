
import argparse
import torch


from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import keras
import tensorflow as tf
from imutils import paths
from mask_generator import my_image_mask_generator
from data_preprocessing import DataPreprocessing
from torch.utils.data import DataLoader
from model import build_unet

def train_loop(args):
    if(args.device=='cuda'):
        config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
        sess = tf.compat.v1.Session(config=config) 
        keras.backend.set_session(sess)

    path = args.annotation_path
    imagePaths = list()
    maskPaths = list()

    dir_list = os.listdir(path)
    dir_list.sort()
    for individual_dir in dir_list:

        masked_directory_path = args.video_data_path+"/"+individual_dir[0:3]+"/maskedImages/"
        raw_directory_path = args.video_data_path+"/"+individual_dir[0:3]+'/rawImages/'

        tempImage = sorted(list(paths.list_images(masked_directory_path)))
        for tempIm in tempImage:
            imagePaths.append(tempIm)
        tempMask = sorted(list(paths.list_images(raw_directory_path)))
        for tempIm in tempMask:
            maskPaths.append(tempIm)

    split = train_test_split(imagePaths, maskPaths,
                         test_size=args.test_split, random_state=args.random_state)
# unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]


    print("[INFO] saving testing image paths...")
    test_paths = os.path.sep.join([args.output_path, "test_paths.txt"])
    f = open(test_paths, "w")
    f.write("\n".join(testImages))
    f.close()

    SEED = 100

    image_data_generator = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        rotation_range = 10,
        zoom_range = 0.1
    ).flow_from_directory(trainImages, batch_size = args.batch_size, target_size = (args.image_resize_dimensions, args.image_resize_dimensions), seed = SEED)

    mask_data_generator = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        rotation_range = 10,
        zoom_range = 0.1
    ).flow_from_directory(trainMasks, batch_size = args.batch_size, target_size = (args.image_resize_dimensions, args.image_resize_dimensions), seed = SEED)

    my_train_generator = my_image_mask_generator(image_data_generator, mask_data_generator)



    image_data_generator = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        rotation_range = 10,
        zoom_range = 0.1
    ).flow_from_directory(testImages, batch_size = args.batch_size, target_size = (args.image_resize_dimensions, args.image_resize_dimensions), seed = SEED)

    mask_data_generator = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        rotation_range = 10,
        zoom_range = 0.1
    ).flow_from_directory(testMasks, batch_size = args.batch_size, target_size = (args.image_resize_dimensions, args.image_resize_dimensions), seed = SEED)

    my_test_generator = my_image_mask_generator(image_data_generator, mask_data_generator)
    
    model = build_unet((512,512,1), n_classes=1)
    model.compile(optimizer=Adam(learning_rate = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(my_train_generator,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=my_test_generator,
                    validation_steps=800)

    model.save('25epoch_lead_vehicle_segmentation.hdf5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# model.summary()    

def main():
    argparser = argparse.ArgumentParser(
        description='Training Model UNet Lead Vehicle Segmentation')
    argparser.add_argument(
        '--annotation_path',
        metavar='AP',
        default='../train_annotations',
        help='Specifies the path for the training folder with annotations')
    argparser.add_argument(
        '--video_data_path',
        metavar='VDP',
        default='../train_videos',
        help='Specifies the path for training dataset')
    argparser.add_argument(
        '--test_split',
        metavar='TS',
        default=0.15,
        type=float,
        help='Specifies train-test split of the dataset')
    argparser.add_argument(
        '--random_state',
        metavar='RS',
        default=42,
        type=int,
        help='Specifies the shuffling constraint')
    argparser.add_argument(
        '--num_classes',
        metavar='C',
        default=1,
        type=int,
        help='Specifies the number of segmentation classes')
    argparser.add_argument(
        '--learning_rate',
        metavar='LR',
        default=0.001,
        type=float,
        help='Specifies the model learning rate')
    argparser.add_argument(
        '--num_epochs',
        metavar='NE',
        default=40,
        type=int,
        help='Specifies the number of epochs for the model')
    argparser.add_argument(
        '--batch_size',
        metavar='BS',
        default=30,
        type=int,
        help='Specifies the batch size for model training')
    argparser.add_argument(
        '--device',
        metavar='D',
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Specifies device to use for model training')
    argparser.add_argument(
        '--image_resize_dimensions',
        metavar='IRD',
        default=512,
        type=int,
        help='Specifies the dimension of the resized image')
    argparser.add_argument(
        '--output_path',
        metavar='OP',
        default='../output',
        help='Specifies the output directory')

    args = argparser.parse_args()

    print(__doc__)

    try:

        train_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
