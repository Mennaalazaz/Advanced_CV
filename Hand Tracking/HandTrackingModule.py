import cv2
import mediapipe as mp

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConfidence=0.5,trackConfidence=0.5):
        self.mode=mode # to set the mode to static or dynamic
        self.maxHands=maxHands # to set the maximum number of hands to be detected
        self.detectionConfidence=detectionConfidence # to set the minimum detection confidence
        self.trackConfidence=trackConfidence # to set the minimum tracking confidence

        # take the hands module from mediapipe
        self.mpHands=mp.solutions.hands 

        # create an object of the Hands class using the parameters
        self.hands=self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
            ) 
        
        # take the drawing module from mediapipe
        self.mpDraw=mp.solutions.drawing_utils 
        self.connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 0),thickness=1,circle_radius=1)
        self.landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 0, 255),thickness=2,circle_radius=2)


    def findHands(self,img,draw=True):

        # convert the image from BGR (openCV format) to RGB (mediapipe format)
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 

        # process the RGB image to detect and track hands
        self.results=self.hands.process(imgRGB) 

        if self.results.multi_hand_landmarks: # if hand is detected
            for handLms in self.results.multi_hand_landmarks: # to iterate through the detected hands
                if draw:
                    # to draw the landmarks and connections on the hand with specifications
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS,self.landmark_drawing_spec,self.connection_drawing_spec)

        return img


    def findPosition(self,img,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks: # if hand is detected
            myHand=self.results.multi_hand_landmarks[handNo] # get the specified hand
            for id,lm in enumerate(myHand.landmark): # to get the id and the landmark of each point on the hand
                    h,w,c=img.shape 
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    lmList.append([id,cx,cy]) # add the id and the pixel coordinates to the list
                    if draw:
                        cv2.circle(img,(cx,cy),7,(0,0,255),cv2.FILLED)

        return lmList                