import cv2
import mediapipe as mp

class PoseEstimation():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode=mode
        self.upBody= upBody
        self.smooth=smooth
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence

        # Initialize MediaPipe pose module
        self.mpPose=mp.solutions.pose 

        # Get instance of Pose class
        self.pose = self.mpPose.Pose(
                                    static_image_mode=self.mode,
                                    smooth_landmarks=self.smooth,
                                    min_detection_confidence=self.detectionConfidence,
                                    min_tracking_confidence=self.trackConfidence)

        # Initialize MediaPipe draw module
        self.mpDraw=mp.solutions.drawing_utils

    def findPose(self,  img,  draw=True):

        # convert the image from BGR (openCV format) to RGB (mediapipe format)
        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the RGB image to detect and track poses
        self.results= self.pose.process(imgRGB)

        if self.results.pose_landmarks: # if pose detected
                if draw:
                    self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPositions(self,  img,  draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id,pose_lm in enumerate(self.results.pose_landmarks.landmark): # to get the id and the landmark of each pose
                h,w,c=img.shape
                cx,cy= int(pose_lm.x*w), int(pose_lm.y*h)
                lmList.append((id,cx,cy))
                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,0),cv2.FILLED)
        return lmList          



        
