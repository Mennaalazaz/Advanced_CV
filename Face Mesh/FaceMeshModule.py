import cv2
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self,staticMode=False, maxFaces=2, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionConfidence=minDetectionConfidence
        self.minTrackingConfidence=minTrackingConfidence

        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionConfidence,
            min_tracking_confidence=self.minTrackingConfidence
        )
        self.mpDraw=mp.solutions.drawing_utils
        self.drawSpec=self.mpDraw.DrawingSpec(color=(0, 255, 0),thickness=1,circle_radius=1)

    def findFaceMesh(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(imgRGB)

        Faces=[] # Store all detected faces

        if self.results.multi_face_landmarks: 
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec,self.drawSpec)
                    
                face=[] # Store one face's landmarks
                for id, lm in enumerate(faceLms.landmark):
                    h,w,c=img.shape
                    x,y=int(lm.x*w), int(lm.y*h)  # Convert normalized landmark coords to pixel values

                    # show the landmark id on the face
                    cv2.putText(img,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(255,0,0),1)

                    face.append([x,y])  # Save landmark position

                Faces.append(face)    # Add one face's landmarks to list      
        return img,Faces           
        