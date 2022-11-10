import string
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
import numpy as np
import os
from statistics import mode
import matplotlib.image as img


image = img.imread('train_videos/000/disparity_PNG/00000000f.png')
# A = np.fromfile('train_videos/000/disparity/00000000f.raw', dtype='int16')
# image = A.reshape([160,168])
# plt.imshow(image)
# plt.show()
# plt.imshow(image)
# plt.show()
rawfile_path = "train_videos/730/disparity/00000000f.raw"
annotation_path = "train_annotations/000.json"
videoPath = 'train_videos/000/Right.mp4'



with open(rawfile_path, 'rb') as f:
    disparity_image = f.read()


right_image_Width=1000
right_image_height=420

f = open(annotation_path)

data = json.load(f)

data_required = data['sequence']

global_distance=[]
actual_distance=[]
ratio_change=[]
for indiData in data_required:
    local_distance=[]
    point_coordinate_x=[]
    point_coordinate_y=[]
    for j in range(int(indiData['TgtYPos_LeftUp']),int(indiData['TgtYPos_LeftUp'])+int(indiData['TgtHeight'])):
        for i in range(int(indiData['TgtXPos_LeftUp']),int(indiData['TgtXPos_LeftUp'])+int(indiData['TgtWidth'])):
            disparity_j = int((right_image_height - j - 1) / 4)  # y-coordinate
            disparity_i = int(i / 4)  # x-coordinate

            
            # Load the disparity map
            # print((disparity_j * 256 + disparity_i) * 2)
            disparity =  disparity_image[(disparity_j * 256 + disparity_i) * 2] # integer
            
            disparity += disparity_image[(disparity_j * 256 + disparity_i) * 2 + 1] / 256 # decimal
            
            if disparity > 0:  # no distance if disparity = 0
                distance=(560 / (disparity - indiData['inf_DP'])) # inf_DP: Infinite disparity
                if(distance <  indiData['Distance_ref'] +2 and distance > indiData['Distance_ref'] -2 ):
                    print("Cal Distance "+str(distance)+" actual distance "+str(indiData['Distance_ref'])+" for pixel "+ str(i) + " and "+str(j))
                    point_coordinate_x.append(i)
                    point_coordinate_y.append(j)

    cap = cv2.VideoCapture(videoPath)
  
    if(cap.isOpened()==False):
        print("Error opening video stream or file")
    i=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            # bounding box for leading vehicle
            # start_point = (int(data['sequence'][i]['TgtXPos_LeftUp']),int(data['sequence'][i]['TgtYPos_LeftUp']))
            # end_point = (int(data['sequence'][i]['TgtXPos_LeftUp']+data['sequence'][i]['TgtWidth']),int(data['sequence'][i]['TgtYPos_LeftUp']+data['sequence'][i]['TgtHeight']))
            color = (255,0,0)
            for j in range(len(point_coordinate_y)):
                frame = cv2.circle(frame, (point_coordinate_x[j],point_coordinate_y[j]),0,color,1)


            cv2.imshow('Frame',frame)

            if cv2.waitKey(25)& 0xFF == ord('q'):
                break
        else:
            break
        i+=1

    cap.release()

    cv2.destroyAllWindows()
#     global_distance.append(sum(local_distance)/len(local_distance))
#     actual_distance.append(indiData['Distance_ref'])            
#     ratio_change.append(indiData['Distance_ref']/(sum(local_distance)/len(local_distance)))
# print(global_distance)



 


    




