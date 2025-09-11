import cv2 
import time
import os
import handTrackingModule  as HTM 

cap= cv2.VideoCapture(2)

# setting size 
wCam , hCam=  640,480
cap.set(3,wCam)
cap.set(4,hCam)

prev_time=0

detector=HTM.HandDetector(detectionConfidence=0.75, trackConfidence=0.75)

# storing finger count pictures as a images
filepath="FingersPic"
myList=os.listdir(filepath) # ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg']
overlayList=[] 
for imPath in myList:
    image=cv2.imread(f"{filepath}/{imPath}")
    image_resized= cv2.resize(image, (160, 120)) # resize the image before assigning to fit the screen
    overlayList.append(image_resized)   # [FingersPic/1.jpg, FingersPic/2.jpg, etc]


def FingersUP (lmList,handType):
    FingersUp=[]
    upCount=0

    if len(lmList)>0: # hand detected
        # for thumb: compare x
        if handType=='Left' : FingersUp.append(1 if lmList[4][1]>lmList[3][1] else 0 ) 
        else: FingersUp.append(1 if lmList[4][1]<lmList[3][1] else 0 )    

        # for rest fingers : compare y
        for id in [8,12,16,20]: FingersUp.append(1 if lmList[id][2]<lmList[id-2][2] else 0 )

        upCount= FingersUp.count(1)    
    return upCount ,  FingersUp
    

while True:
    success, img= cap.read()

    img=detector.findHands(img)
    lmList,handType=detector.findPosition(img)

    upCount,FingersUp=FingersUP(lmList,handType)

    print(FingersUp)

    # add the overlay image on the screen 
    h,w,c=overlayList[upCount-1].shape
    img[0:h,0:w]=overlayList[upCount-1]

    # show the upCount on the screen
    cv2.rectangle(img,(0,200),(160,480),(255,0,0),cv2.FILLED)
    cv2.putText(img,str(upCount),(50,370),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,3,(0,255,0),3)

    # fps (frame per second)
    curr_time=time.time()
    fps=1/(curr_time-prev_time)
    prev_time=curr_time
    cv2.putText(img,f"FPS:{int(fps)}",(400,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)