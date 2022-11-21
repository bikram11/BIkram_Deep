from arcgis.learn import SiamMask
import numpy as np
import cv2
import json
import pandas as pd

ot = SiamMask()


cap = cv2.VideoCapture(r"../test_videos/240/Right.mp4")

masked_image_path="../test_videos/240/masked_files/" 
from pathlib import Path
Path(masked_image_path).mkdir(parents=True, exist_ok=True)

annotation_path = "../test_annotations/240.json"
f = open(annotation_path)
data = json.load(f)
data_required = data['sequence']

df_annotation_list=pd.json_normalize(data, record_path=['sequence'])

print(df_annotation_list['TgtXPos_LeftUp'][0])