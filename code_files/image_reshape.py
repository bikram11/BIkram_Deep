import cv2
import numpy as np









import cv2
import json
import numpy as np
import os
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


path = '../train_annotations'

dir_list = os.listdir(path)
dir_list.sort()
for inddir in dir_list:
    if(int(inddir[0:3])>418):
   
        maskedImagePath = '../train_videos/'+inddir[0:3]+"/maskedImages/"
        rawImagePath = '../train_videos/'+inddir[0:3]+"/rawImages/"

        # image_names = glob.glob(rawImagePath+"*.png")
        # image_names.sort()
        # for i in range(len(image_names)):
        #     im = Image.open(image_names[i])
        #     im_resize = im.resize((512,512))
        #     im_resize.save(image_names[i])



        image_names = glob.glob(maskedImagePath+"*.png")
        image_names.sort()
        for i in range(len(image_names)):
            print(image_names[i])
            im = Image.open(image_names[i])
            im_resize = im.resize((512,512))
            im_resize.save(image_names[i])
