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
image = img.imread('train_videos/000/disparity_PNG/00000000f.png')
# A = np.fromfile('train_videos/000/disparity/00000000f.raw', dtype='int16')
# image = A.reshape([160,168])
# plt.imshow(image)
# plt.show()
# plt.imshow(image)
# plt.show()
rawfile_path = "train_videos/720/disparity/00000000f.raw"
annotation_path = "train_annotations/720.json"
videoPath = 'train_videos/720/Right.mp4'



with open(rawfile_path, 'rb') as f:
    disparity_image = f.read()


right_image_Width=1000
right_image_height=420

f = open(annotation_path)

data = json.load(f)

data_required = data['sequence']

global_distance=[]
estimated_velocity=[]
actual_distance=[]
ratio_change=[]
n_bins= 10
for indiData in data_required:
    local_distance=[]
    distance_change=0
    pre_distance=0
    for j in range(int(indiData['TgtYPos_LeftUp'])+10,int(indiData['TgtYPos_LeftUp'])+int(indiData['TgtHeight'])-10):
        for i in range(int(indiData['TgtXPos_LeftUp'])+10,int(indiData['TgtXPos_LeftUp'])+int(indiData['TgtWidth'])-10):
            disparity_j = int((right_image_height - j - 1) / 4)  # y-coordinate
            disparity_i = int(i / 4)  # x-coordinate

            
            # Load the disparity map
            # print((disparity_j * 256 + disparity_i) * 2)
            disparity =  disparity_image[(disparity_j * 256 + disparity_i) * 2] # integer
            
            disparity += disparity_image[(disparity_j * 256 + disparity_i) * 2 + 1] / 256 # decimal
            
            if disparity > 0: 
                distance =  560 / (disparity - indiData['inf_DP'])
                
                if distance > 0 and distance < 150 :# no distance if disparity = 0
                    local_distance.append(560 / (disparity - indiData['inf_DP']))
                    # inf_DP: Infinite disparity
                # if(distance <  indiData['Distance_ref'] +5 and distance > indiData['Distance_ref'] -5 ):
                #     print("Cal Distance "+str(distance)+" actual distance "+str(indiData['Distance_ref'])+" for pixel "+ str(i) + " and "+str(j))
    # fig, axs = plt.subplots(1,1, figsize=(10,7), tight_layout=True)
    # scipy_kernel = gaussian_kde(local_distance)
    # u= np.linspace(0,150,500)
    # v=scipy_kernel.evaluate(u)
    # # axs.hist(local_distance,bins=n_bins)
    # plt.plot(u,v,'k')
    local_distance1 = np.array(local_distance)
    local_distance1.shape = (local_distance1.size,1)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(local_distance1)
    sample = kde.sample(5000)
    c,b,p = plt.hist(sample,bins=100, edgecolor = 'black')
   
    distance_DataFrame = pd.Series(local_distance)

    bins = np.arange(distance_DataFrame.min(), distance_DataFrame.max()+2)

    h, _ = np.histogram(sample,bins)

    modei = bins[h.argmax()]

    mode_pd = distance_DataFrame.value_counts().nlargest(1).index[0]

    print("Calculated with frequent value from histogram: "+str(modei)+" Calculated with pandas from histogram: "+str(mode_pd)+" Actual Value: "+str(indiData['Distance_ref'])+ " General Mean: "+str(mean(local_distance))+ " General Median: "+str(median(local_distance))+ " General Mode: "+str(mode(local_distance)))
    
    # plt.show()    
    global_distance.append(mean(local_distance))
    if(pre_distance==0):
        speed = (mean(local_distance)-int(mean(local_distance)))/0.1
        estimated_velocity.append(indiData['OwnSpeed']+speed)
        pre_distance=mean(local_distance)
    else:
        speed =(mean(local_distance)-pre_distance)/0.1
        estimated_velocity.append(indiData['OwnSpeed']+speed)
        pre_distance=mean(local_distance)
    
    actual_distance.append(indiData['Distance_ref'])            
    ratio_change.append(indiData['Distance_ref']/(sum(local_distance)/len(local_distance)))




 


    




