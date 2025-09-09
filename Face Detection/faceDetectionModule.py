import cv2
import mediapipe as mp

class FaceDetection():
    def __init__(self,minDetectionConfidence=0.5):
        self.minDetectionConfidence=minDetectionConfidence

        self.mpFaceDetection = mp.solutions.face_detection # Get the Module
        self.faceDetection= self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionConfidence) # Get an object of the class
        self.mpDraw= mp.solutions.drawing_utils  

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results.detections)

        bboxs = [] # store bounding boxes

        if self.results.detections: # if face is detected
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape

                # convert relative to pixel coords
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h) 

                bbox = (x, y, width, height)
                bboxs.append(bbox)

                if draw:
                    img=self.fancyDraw(img,bbox)
                    # Adding the score 
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (x, y-10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, bboxs
    

    def fancyDraw(self,img,bbox, lineLength=30, thickness=10):
        x,y,w,h=bbox
        x1,y1=x+w, y+h
        cv2.rectangle(img, (x, y), (x1,y1), (255, 0, 255), 2)
        # top left (x,y)
        cv2.line(img,(x,y),(x+lineLength,y),(255, 0, 255),thickness)
        cv2.line(img,(x,y),(x,y+lineLength),(255, 0, 255),thickness)

        # top Right (x1,y)
        cv2.line(img,(x1,y),(x1-lineLength,y),(255, 0, 255),thickness)
        cv2.line(img,(x1,y),(x1,y+lineLength),(255, 0, 255),thickness)

        # bottom left (x,y1)
        cv2.line(img,(x,y1),(x+lineLength,y1),(255, 0, 255),thickness)
        cv2.line(img,(x,y1),(x,y1-lineLength),(255, 0, 255),thickness)

        # bottom Right (x1,y1)
        cv2.line(img,(x1,y1),(x1-lineLength,y1),(255, 0, 255),thickness)
        cv2.line(img,(x1,y1),(x1,y1-lineLength),(255, 0, 255),thickness)

        return img



# When you run MediaPipe Face Detection, each detection object (detection) contains several pieces of info:
    # score → confidence the object is a face
    # label_id → class id (for face detection, usually just 0)
    # location_data → where the object is located in the image
        # Inside location_data, you can choose formats:
        # format: RELATIVE_BOUNDING_BOX → bounding box is expressed in relative (normalized) coordinates between 0 and 1
            # So when you access:   bboxC = detection.location_data.relative_bounding_box
            # You get an object that has 4 fields:

            # bboxC.xmin → left edge of the face box (relative to image width)

            # bboxC.ymin → top edge of the face box (relative to image height)

            # bboxC.width → box width (relative to image width)

            # bboxC.height → box height (relative to image height)




        