# importing the zipfile module
from zipfile import ZipFile

import argparse

def extraction_loop(args):
# loading the temp.zip and creating a zip object
    with ZipFile(args.zip_file_path, 'r') as zObject:
    
        # Extracting specific file in the zip
        # into a specific location.
        zObject.extractall(
            path=args.destination_path)
    zObject.close()


def main():
    argparser = argparse.ArgumentParser(
        description='Zip file Extraction')
    argparser.add_argument(
        '--zip_file_path',
        metavar='ZFP',
        default='train_videos_1.zip',
        help='Specifies the path for zip folder that needs to be extracted')
    argparser.add_argument(
        '--destination_path',
        metavar='DP',
        default='training',
        help='Specifies the folder destination to store the content of zip file')

    args = argparser.parse_args()


    try:

        extraction_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
