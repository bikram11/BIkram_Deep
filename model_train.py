import string
from threading import local
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
import numpy as np
from tqdm import tqdm
import os
from statistics import median, mode,mean
import matplotlib.image as img
from matplotlib.ticker import PercentFormatter
from matplotlib import colors
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow import keras
import scipy.stats as sts
from data_preprocessing.data_loader import DataLoader
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import argparse
import pickle


def train_lightgbm_model(args):
    annotation_path = args.annotation_path
    dir_list = os.listdir(annotation_path)
    dir_list.sort()
    df = pd.DataFrame()
    for individual_annotation in tqdm(dir_list,desc='dirs'):
        f = open(annotation_path+individual_annotation)
        data = json.load(f)
        local_distance = DataLoader(data['sequence'],args.backtrack_coefficient)
        df=pd.concat([df,local_distance])


    Y=df['TgtSpeed_ref']
    X=df.drop(columns=['TgtSpeed_ref','inf_DP','TgtXPos_LeftUp','TgtYPos_LeftUp','TgtWidth','TgtHeight'])

    x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=args.split_coefficient,shuffle=True)


    params={

        'task':'train',
        'boosting':args.boosting,
        'objective':'regression',
        'num_leaves':args.num_leaves,
        'learning_rate':args.learning_rate,
        'metric':{'l2','l1'},
        'verbose':-1,
        'num_boost_round':args.num_boost_round,
        'early_stopping_round':args.early_stopping_round
    }


    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_eval=lgb.Dataset(x_test,y_test, reference=lgb_train)

    model = lgb.train(params,train_set=lgb_train, valid_sets=lgb_eval)

    pickle.dump(model, open(args.model_location+"_"+args.boosting+"_"+args.num_leaves+"_"+args.learning_rate+"_"+args.early_stopping_round+".sav", 'wb'))


def main():
    argparser = argparse.ArgumentParser(
        description='Image and Mask Frame Generation')
    argparser.add_argument(
        '--annotation_path',
        metavar='AP',
        default='train_annotations/',
        help='Specifies the path for the training/test folder with annotations')
    argparser.add_argument(
        '--model_location',
        metavar='ML',
        default='LightGBMV1',
        help='Specifies the path to save the final model')
    argparser.add_argument(
        '--backtrack_coefficient',
        metavar='BC',
        default=18,
        type=int,
        help='Specifies the number of past time frames to include for the current prediction')
    argparser.add_argument(
        '--split_coefficient',
        metavar='SC',
        default=0.2,
        type=float,
        help='Specifies the ratio of train-test-split')
    argparser.add_argument(
        '--learning_rate',
        metavar='LR',
        default=0.05,
        type=float,
        help='Specifies the learning rate for the model')
    argparser.add_argument(
        '--boosting',
        default="dart",
        help='Specifies the gradient boosting to use with lightGBM, choose between [gbdt, goss, dart, rf]')
    argparser.add_argument(
        '--num_leaves',
        default=70,
        type=int,
        help='Specifies the number of leaves node allowed in the Decision tree')
    argparser.add_argument(
        '--num_boost_round',
        metavar='NBR',
        default=5000,
        type=int,
        help='Specifies the number of boost rounds the model should run')
    argparser.add_argument(
        '--early_stopping_round',
        metavar='ESR',
        default=30,
        type=int,
        help='Specifies the early stopping condition')

    args = argparser.parse_args()

    try:

        train_lightgbm_model(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()