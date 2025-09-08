import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(2) # to capture video from webcam 

mpHands=mp.solutions.hands # to detect and track hands using mediapipe
hands=mpHands.Hands() # to create an object of the Hands class
mpDraw=mp.solutions.drawing_utils # to draw the landmarks and connections on the hand


prevTime=0
currTime=0


while True:     # make the loop infinite to read the video frame by frame
        success,img=cap.read() # read the image frame

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # convert the image from BGR (openCV format) to RGB (mediapipe format)
        results=hands.process(imgRGB) # process the RGB image to detect and track hands

        # to print the landmarks of the detected hands
        print("landmarks",results.multi_hand_landmarks)

        if results.multi_hand_landmarks: # if hand is detected
            for handLms in results.multi_hand_landmarks: # to iterate through the detected hands
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) # to draw the landmarks and connections on the hand
                for id,lm in enumerate(handLms.landmark): # to get the id and the landmark of each point on the hand
                    h,w,c=img.shape # to get the height, width and channels of the image
                    cx,cy=int(lm.x*w),int(lm.y*h) # to convert the normalized landmark coordinates to pixel coordinates
                    print("pixel coordinates",id,cx,cy)

                    # to draw a circle on the tip of thumb (id=4)
                    if id==4:
                        cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)


        # to calculate the frames per second (fps)
        currTime=time.time()
        fps=1/(currTime-prevTime)
        prevTime=currTime  

        # to display the fps on the image
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)             
        

        cv2.imshow("Image",img)
        cv2.waitKey(1)  # wait for 1 millisecond before moving to next frame



