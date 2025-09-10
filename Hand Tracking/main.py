import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

def main():
    cap=cv2.VideoCapture(2) # to capture video from webcam 

    detector=htm.handDetector() # create an object of the handDetector class

    prevTime=0
    currTime=0

    while True:     # make the loop infinite to read the video frame by frame
        success,img=cap.read() # read the image frame

        img=detector.findHands(img) # find and draw the hands

        lmList=detector.findPosition(img) # find the position of the landmarks
        if len(lmList)!=0: 
            print(lmList[4]) # print the position of the tip of the thumb (id=4)

        # to calculate the frames per second (fps)
        currTime=time.time()
        fps=1/(currTime-prevTime)
        prevTime=currTime  

        # to display the fps on the image
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)             
        

        cv2.imshow("Image",img)
        cv2.waitKey(1)  # wait for 1 millisecond before moving to next frame    


if __name__=="__main__":
    main()