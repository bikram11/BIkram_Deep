import cv2
import numpy as np









import cv2
import json
import numpy as np
import os

path = '../train_annotations'

dir_list = os.listdir(path)
dir_list.sort()
# for inddir in dir_list:
if(1==1):
    # print(inddir[0:3])
    # videoPath = '../train_videos/'+inddir[0:3]+'/Right.mp4'
    videoPath = '../train_videos/419/Right.mp4'
    jsonPath = path+"/"+"419.json"
    maskedImagePath = '../train_videos/419/maskedImages/'
    rawImagePath = '../train_videos/419/rawImages/'
    cap = cv2.VideoCapture(videoPath)





    f = open(jsonPath)

    data = json.load(f)

    if(cap.isOpened()==False):
        print("Error opening video stream or file")
    i=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            # bounding box for leading vehicle
            start_x_point = 0 if int(data['sequence'][i]['TgtXPos_LeftUp']) < 0 else int(data['sequence'][i]['TgtXPos_LeftUp'])
            start_y_point = 0 if int(data['sequence'][i]['TgtYPos_LeftUp']) < 0 else int(data['sequence'][i]['TgtYPos_LeftUp'])
            start_point = (start_x_point,start_y_point)
            end_point = (int(start_x_point+data['sequence'][i]['TgtWidth']),int(start_y_point+data['sequence'][i]['TgtHeight']))
            color = (255,0,0)
            # img = cv2.imread('image1.png') # read image
            mask = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8) # initialize mask
            mask[start_point[1]:end_point[1],start_point[0]:end_point[0]] = 255 # fill with white pixels
            print(maskedImagePath+str(i)+'_mask.png')
            cv2.imwrite(maskedImagePath+str(i)+'_mask.png',mask) # save mask
            # frame = cv2.rectangle(frame, start_point,end_point,color,2)


            # text = 'EgoVechile speed: %s'%(data['sequence'][i]['OwnSpeed'])
            # frame = cv2.putText(frame, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            # text = 'Steering Degree : %s'%(data['sequence'][i]['StrDeg'])
            # frame = cv2.putText(frame, text, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            # text = 'Leading Vehicle Distance : %s'%(data['sequence'][i]['Distance_ref'])
            # frame = cv2.putText(frame, text, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            # text = 'Leading  Vehicle Speed : %s'%(data['sequence'][i]['TgtSpeed_ref'] )
            # frame = cv2.putText(frame, text, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            i+=1
            cv2.imshow('Frame',frame)

            if cv2.waitKey(25)& 0xFF == ord('q'):
                break
        else:
            break


    cap.release()

    cv2.destroyAllWindows()