from arcgis.learn import SiamMask
import numpy as np
import cv2
import json
import pandas as pd
import os
ot = SiamMask()
path = 'test_annotations/'

dir_list = os.listdir(path)
dir_list.sort()
for inddir in dir_list:

    cap = cv2.VideoCapture("test_videos/"+inddir[0:3]+"/Right.mp4")

    # masked_image_path="../test_videos/"+inddir[0:3]+"/masked_files/" 
    # from pathlib import Path
    # Path(masked_image_path).mkdir(parents=True, exist_ok=True)

    annotation_path = path+inddir
    f = open(annotation_path)
    data = json.load(f)
    data_required = data['sequence']
    df_annotation_list=pd.json_normalize(data, record_path=['sequence'])

    from PIL import Image

    initialized = False
    i=0
    while(True):
        ret, frame = cap.read()
        if ret is False:
            break
        lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        if initialized:
            state = ot.update(enhanced_img)  ## Update the track location in the frame
            for track in state:
                mask = track.mask
                enhanced_img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * enhanced_img[:, :, 2]
                # mask = img_grey(mask)
                # mask = img_frombytes(mask)
                # bbox = extract_bboxes(np.array(mask))
                # print(bbox)
                # mask = mask.save(masked_image_path+str(i) +
                #                 '_mask.png')
                # cv2.imwrite(masked_image_path+str(i) +
                #                 '_mask.png', mask) 
                # frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
                abc = np.int0(track.location).reshape((-1, 1, 2))
                a1=abc.max(axis=0,keepdims=True)
                a2=abc.min(axis=0,keepdims=True)

                a3=a1-a2
                df_annotation_list['TgtXPos_LeftUp'][i]=0.0 if a2[0][0][0] < 0 else a2[0][0][0]
                df_annotation_list['TgtYPos_LeftUp'][i]=0.0 if a2[0][0][1] < 0 else a2[0][0][1]
                df_annotation_list['TgtWidth'][i]=0.0 if a3[0][0][0] < 0 else a3[0][0][0]
                df_annotation_list['TgtHeight'][i]=0.0 if a3[0][0][1] < 0 else a3[0][0][1]

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
    Path("../complete_test_annotations_2/").mkdir(parents=True, exist_ok=True)
    result = df_annotation_list.to_json('../complete_test_annotations_2/'+inddir,orient='records')
