import cv2
import time
import math
import numpy as np
import HandTrackingModule as HTM
from volume_control import VolumeController
from media_control import MediaController

cap = cv2.VideoCapture(2)

prevTime=0
detector=HTM.HandDetector(detectionConfidence=0.7)
volume=VolumeController()
media=MediaController()


while True:
    success,img= cap.read()

    img=detector.findHands(img)
    lmList,handType=detector.findPosition(img,draw=False)

    if len(lmList):
        # Right hand → volume control , Left hand → media control

        if handType=='Right':
            # CoolDown to prevent repeated actions
            if time.time() - volume.last_action > volume.coolDown:
            
                #The landmarks wanna track are thumb tip & index finger 
                #print(lmList[4],lmList[8])

                # unpack their positions and calculate the point of median (cx,cy)
                x1, y1= lmList[4][1],  lmList[4][2]
                x2, y2= lmList[8][1],  lmList[8][2]
                cx, cy= (x1+x2)//2 , (y1+y2)//2
            
                # Add circles on thumb tip , index finger tip & point of median
                cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
                cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
                cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                # Add line between them
                cv2.line(img,(x1,y1), (x2,y2),(255,0,255),3)

                # Calculate the distance between (x1,y1) and (x2,y2)
                length=math.dist([x1,y1],[x2,y2])

                # Map hand length (20–300) → volume percentage (0–100)
                volPercent = np.interp(length, [20, 300], [0, 100])
                volume.setCurrentVolume(volPercent)

                # if fingers are close , the median circle will be colored green
                if length < 50:
                    cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

                volume.last_action=time.time()   
        else :
            # control media (play/pause) 
            Fingers=media.isFingerUP(lmList)
            media.controlMedia(Fingers)



    # Handle the case when you change the volume using the mouse or keyboard (No Hand detected)
    vol=volume.getCurrentVolume() 
    volPercent = volume.getVolumePercent(vol)
    volBar = volume.getVolumeBar(volPercent)
        
    # Draw volume bar
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3) # Empty Rectangle
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED) # Filled Rectangle
    cv2.putText(img,f'{int(volPercent)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX ,1,(0,255,0),3)

    # FPS
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime  
    cv2.putText(img,f'FPS:{int(fps)}',(30,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)             

    cv2.imshow('Image',img)
    cv2.waitKey(1)