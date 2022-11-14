# import the necessary packages
from torch.utils.data import Dataset
import cv2
import glob
import numpy as np


class DataPreprocessing(Dataset):
    def __init__(self, imagePaths, maskPaths):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths


    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.maskPaths[idx], 0)
        # check to see if we are applying any transformations
            # apply the transformations to both image and its mask
        image = image/255
        mask = mask/255
        # return a tuple of the image and its mask
        return (image, mask)
