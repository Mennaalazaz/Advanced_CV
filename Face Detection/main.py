import cv2
import time
import faceDetectionModule as FDM

def main():
    cap= cv2.VideoCapture('video/faces.mp4')

    detector= FDM.FaceDetection()

    prev_time=0


    while True:
        success, img=cap.read()


        # show frames per second (fps)
        curr_time=time.time()
        fps= 1/(curr_time-prev_time)
        prev_time=curr_time
        cv2.putText(img, f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        # Detect faces
        img, bboxs = detector.findFaces(img)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()