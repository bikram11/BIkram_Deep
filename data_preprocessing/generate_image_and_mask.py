import cv2
import numpy as np


import cv2
import json
import numpy as np
import os
import argparse


def extract_frame_loop(args):
    path = args.annotation_path


    dir_list = os.listdir(path)
    dir_list.sort()
    for individual_dir in dir_list:

        video_path = args.video_path+"/"+individual_dir[0:3]+'/Right.mp4'
        json_path = path+"/"+individual_dir[0:3]+".json"
        masked_image_path = args.video_path+"/"+individual_dir[0:3]+"/maskedImages/"
        raw_image_path = args.video_path+"/"+individual_dir[0:3]+'/rawImages/'
        cap = cv2.VideoCapture(video_path)
    
        f = open(json_path)

        data = json.load(f)

        if(cap.isOpened() == False):
            print("Error opening video stream or file")
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # bounding box for leading vehicle
                start_x_point = 0 if int(data['sequence'][i]['TgtXPos_LeftUp']) < 0 else int(
                    data['sequence'][i]['TgtXPos_LeftUp'])
                start_y_point = 0 if int(data['sequence'][i]['TgtYPos_LeftUp']) < 0 else int(
                    data['sequence'][i]['TgtYPos_LeftUp'])
                start_point = (start_x_point, start_y_point)
                end_point = (int(start_x_point+data['sequence'][i]['TgtWidth']), int(
                    start_y_point+data['sequence'][i]['TgtHeight']))
                # initialize mask
                mask = np.zeros(
                    (frame.shape[0], frame.shape[1]), dtype=np.uint8)
                mask[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 255  # fill with white pixels
                print(masked_image_path+str(i)+'_mask.png')

                frame = cv2.resize(frame, (args.final_image_dimension, args.final_image_dimension)) 
                mask = cv2.resize(mask, (args.final_image_dimension, args.final_image_dimension)) 
                cv2.imwrite(masked_image_path+str(i) +
                            '_mask.png', mask)  # save mask
                cv2.imwrite(raw_image_path+str(i)+'_raw.png', frame)  # save mask
                i += 1

            else:
                break

        cap.release()


def main():
    argparser = argparse.ArgumentParser(
        description='Image and Mask Frame Generation')
    argparser.add_argument(
        '--annotation_path',
        metavar='AP',
        default='../train_annotations',
        help='Specifies the path for the training folder with annotations')
    argparser.add_argument(
        '--video_path',
        metavar='VP',
        default='../train_videos',
        help='Specifies the path for the training video root directory')
    argparser.add_argument(
        '--final_image_dimension',
        metavar='VP',
        default=512,
        type=int,
        help='Specifies the final dimension of the image frame you want to save')

    args = argparser.parse_args()

    try:

        extract_frame_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
