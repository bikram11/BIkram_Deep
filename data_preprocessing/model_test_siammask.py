from arcgis.learn import SiamMask
import numpy as np
import cv2
import json
import pandas as pd
import os
import argparse

from PIL import Image
from PIL import ImageEnhance


ot = SiamMask()

def extract_mask_from_test_video(args):
    path = args.annotation_path

    dir_list = os.listdir(path)
    dir_list.sort()
    for individual_video in dir_list:
        # read the video frame
        cap = cv2.VideoCapture(args.video_path+individual_video[0:3]+"/Right.mp4")


        # read the annotation frame
        annotation_path = path+individual_video
        f = open(annotation_path)
        data = json.load(f)
        data_required = data['sequence']
        df_annotation_list=pd.json_normalize(data, record_path=['sequence'])


        # check if the siam mask has been initialized with the first frame
        initialized = False
        i=0
        while(True):
            ret, frame = cap.read()
            if ret is False:
                break
            if(args.image_enhancement_on):
                image_frame=Image.fromarray(frame)
                enhancer_sharpness = ImageEnhance.Sharpness(image_frame).enhance(args.sharpness_coefficient)
                enhancer_color = ImageEnhance.Color(enhancer_sharpness).enhance(args.color_coefficient)

                enhanced_img=np.array(enhancer_color)
            else:
                enhanced_img=frame


            if initialized:
                state = ot.update(enhanced_img)  ## Update the track location in the frame
                for track in state:
                    #get the mask
                    mask = track.mask
                    enhanced_img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * enhanced_img[:, :, 2]

                    #reshape the mask bounding box to matrix
                    mask_matrix = np.int0(track.location).reshape((-1, 1, 2))

                    # get the max and min component of the bounding box
                    bounding_box_max=mask_matrix.max(axis=0,keepdims=True)
                    bounding_box_min=mask_matrix.min(axis=0,keepdims=True)

                    bounding_box_width_heigth=bounding_box_max-bounding_box_min
                    df_annotation_list['TgtXPos_LeftUp'][i]=0.0 if bounding_box_min[0][0][0] < 0 else bounding_box_min[0][0][0]
                    df_annotation_list['TgtYPos_LeftUp'][i]=0.0 if bounding_box_min[0][0][1] < 0 else bounding_box_min[0][0][1]
                    df_annotation_list['TgtWidth'][i]=0.0 if bounding_box_width_heigth[0][0][0] < 0 else bounding_box_width_heigth[0][0][0]
                    df_annotation_list['TgtHeight'][i]=0.0 if bounding_box_width_heigth[0][0][1] < 0 else bounding_box_width_heigth[0][0][1]

                    cv2.polylines(enhanced_img, [np.int0(track.location).reshape((-1, 1, 2))], True, (w, 255, h), 1)

            cv2.imshow('frame',enhanced_img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            else:
                data_required[0]['TgtXPos_LeftUp'] = 0 if int(data_required[0]['TgtXPos_LeftUp']) < 0 else int(data_required[0]['TgtXPos_LeftUp'])
                data_required[0]['TgtYPos_LeftUp'] = 0 if int(data_required[0]['TgtYPos_LeftUp']) < 0 else int(data_required[0]['TgtYPos_LeftUp'])
                x=int(data_required[0]['TgtXPos_LeftUp'])
                y= int(data_required[0]['TgtYPos_LeftUp'])
                w=int(data_required[0]['TgtWidth'])
                h=int(data_required[0]['TgtHeight'])
                state = ot.init(enhanced_img, [[x,y,w,h]]) ## Initialize the track in the frame
                initialized = True
            i+=1
        cap.release()
        cv2.destroyAllWindows()
        from pathlib import Path
        Path(args.output_annotation_path).mkdir(parents=True, exist_ok=True)
        result = df_annotation_list.to_json(args.output_annotation_path+individual_video,orient='records')



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
        '--image_enhancement_on',
        metavar='IE',
        default=True,
        type=bool,
        help='Specifies whether to enhance image for the final prediction or not')
    argparser.add_argument(
        '--sharpness_coefficient',
        metavar='SC',
        default=2,
        type=int,
        help='Specifies the sharpness for the image enhancement')
    argparser.add_argument(
        '--color_coefficient',
        metavar='CC',
        default=0.8,
        type=float,
        help='Specifies the color change for the color enhancement')

    args = argparser.parse_args()

    try:

        extract_mask_from_test_video(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()