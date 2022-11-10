import cv2
import json
import numpy as np
import os

path = 'train_annotations'

dir_list = os.listdir(path)

for inddir in dir_list:
    
    videoPath = 'train_videos/'+inddir[0:3]+'/Right.mp4'
    jsonPath = path+"/"+inddir

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
            start_point = (int(data['sequence'][i]['TgtXPos_LeftUp']),int(data['sequence'][i]['TgtYPos_LeftUp']))
            end_point = (int(data['sequence'][i]['TgtXPos_LeftUp']+data['sequence'][i]['TgtWidth']),int(data['sequence'][i]['TgtYPos_LeftUp']+data['sequence'][i]['TgtHeight']))
            color = (255,0,0)
            frame = cv2.rectangle(frame, start_point,end_point,color,2)


            text = 'EgoVechile speed: %s'%(data['sequence'][i]['OwnSpeed'])
            frame = cv2.putText(frame, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            text = 'Steering Degree : %s'%(data['sequence'][i]['StrDeg'])
            frame = cv2.putText(frame, text, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            text = 'Leading Vehicle Distance : %s'%(data['sequence'][i]['Distance_ref'])
            frame = cv2.putText(frame, text, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )

            text = 'Leading  Vehicle Speed : %s'%(data['sequence'][i]['TgtSpeed_ref'] )
            frame = cv2.putText(frame, text, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA )


            cv2.imshow('Frame',frame)

            if cv2.waitKey(25)& 0xFF == ord('q'):
                break
        else:
            break
        i+=1

    cap.release()

    cv2.destroyAllWindows()