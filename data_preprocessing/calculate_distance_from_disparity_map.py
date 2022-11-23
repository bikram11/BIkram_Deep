import string
from threading import local
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
import numpy as np
import os
from statistics import median, mode,mean
import matplotlib.image as img
from matplotlib.ticker import PercentFormatter
from matplotlib import colors
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import scipy.stats as sts
from tqdm import tqdm
import argparse


def return_distance_ref(rawfile_path,raw_file_list,data_required,right_image_height):
    distance_ref=[]
    for x in range(len(data_required)):

        with open(rawfile_path+raw_file_list[x], 'rb') as f:
            disparity_image = f.read()
        local_distance=[]
    
        for j in range(int(data_required[x]['TgtYPos_LeftUp']),int(data_required[x]['TgtYPos_LeftUp'])+int(data_required[x]['TgtHeight'])):
            for i in range(int(data_required[x]['TgtXPos_LeftUp']),int(data_required[x]['TgtXPos_LeftUp'])+int(data_required[x]['TgtWidth'])):
                disparity_j = int((right_image_height - j - 1) / 4)  # y-coordinate
                disparity_i = int(i / 4)  # x-coordinate
                # Load the disparity map
                disparity =  disparity_image[(disparity_j * 256 + disparity_i) * 2] # integer

                disparity += disparity_image[(disparity_j * 256 + disparity_i) * 2 + 1] / 256 # decimal

                if disparity > 0: 
                    distance =  560 / (disparity - data_required[x]['inf_DP'])
                    local_distance.append(distance)

        # plot the real KDE
        kde = sts.gaussian_kde(local_distance)


        # plot the KDE
        height = kde.pdf(local_distance)
        mode_value = local_distance[np.argmax(height)]
        distance_ref.append(mode_value)
    return distance_ref



def extract_distance_from_disparity(args):
    annotation_path = args.annotation_path
    dir_list = os.listdir(annotation_path)
    dir_list.sort()
    for individual_video in tqdm(dir_list):
        if(int(individual_video[0:3])>=0):
            f = open(annotation_path+individual_video)
            data = json.load(f)
            df_annotation_list=pd.json_normalize(data)
            disparity_path=args.video_path+individual_video[0:3]+"/disparity/"
            raw_file_list = os.listdir(disparity_path)
            raw_file_list.sort()

            local_distance = return_distance_ref(disparity_path,raw_file_list,data,args.right_image_height)
            df_annotation_list['Distance_ref']=local_distance
            result = df_annotation_list.to_json(args.complete_test_annotations+individual_video,orient='records')


def main():
    argparser = argparse.ArgumentParser(
        description='Image and Mask Frame Generation')
    argparser.add_argument(
        '--annotation_path',
        metavar='AP',
        default='../test_annotations',
        help='Specifies the path for the training/test folder with annotations')
    argparser.add_argument(
        '--video_path',
        metavar='VP',
        default='../test_videos',
        help='Specifies the path for the training/test video root directory')
    argparser.add_argument(
        '--output_annotation_path',
        metavar='VP',
        default='../complete_test_annotations',
        help='Specifies the path for the output folder with complete annotations')
    argparser.add_argument(
        '--right_image_height',
        metavar='RIH',
        default=420,
        type=int,
        help='Specifies the height of the right image in the video')

    args = argparser.parse_args()

    try:

        extract_distance_from_disparity(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()