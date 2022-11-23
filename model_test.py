from tqdm import tqdm
import pandas as pd
import os
from data_preprocessing.data_loader import DataLoader
import pickle
import argparse


def test_lightgbm_model(args):
    annotation_path = args.annotation_path
    dir_list = os.listdir(annotation_path)
    dir_list.sort()
    df1 = pd.DataFrame()
    import json
    frame_id, speed=[],[]
    for individual_annotation in tqdm(dir_list,desc='dirs'):
        f = open(annotation_path+individual_annotation)
        data = json.load(f)
        # print(len(data))
        # print(len(raw_file_list))
        local_distance = DataLoader(data,args.backTrack_coefficient)
        X_pred=local_distance.drop(columns=['inf_DP','TgtXPos_LeftUp','TgtYPos_LeftUp','TgtWidth','TgtHeight'])
        # # print(local_distance)
        model= pickle.load(open(args.model_filename, 'rb'))
        y_test_pred=model.predict(X_pred)
        y_test_pred_decimal=[ round(elem, 1) for elem in y_test_pred ]

        frame_id.append(individual_annotation[0:3])
        speed.append(y_test_pred_decimal)
    final_json=[{t:s} for t,s in zip(frame_id,speed)]
    submission_results=json.dumps(final_json)
    with open(args.submission_file_location,"w") as outfile:
        outfile.write(submission_results)



def main():
    argparser = argparse.ArgumentParser(
        description='Image and Mask Frame Generation')
    argparser.add_argument(
        '--annotation_path',
        metavar='AP',
        default='../complete_test_annotations_2/',
        help='Specifies the path for the test folder with annotations')
    argparser.add_argument(
        '--model_filename',
        metavar='ML',
        default='LightGBMV1.sav',
        help='COmplete filename for the model in testing')
    argparser.add_argument(
        '--backtrack_coefficient',
        metavar='BC',
        default=18,
        type=int,
        help='Specifies the number of past time frames to include for the current prediction')
    argparser.add_argument(
        '--submission_file_location',
        metavar='SC',
        default="./submission.json",
        type=float,
        help='Specifies the submission filename and location')

    args = argparser.parse_args()

    try:

        test_lightgbm_model(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()