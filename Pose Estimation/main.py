import cv2
import time
import PoseModule as PM


def main():

    cap= cv2.VideoCapture('motionVideos/1.mp4')

    # Create an object of PoseEstimation class
    detector=PM.PoseEstimation()

    prev_time=0


    while True:
        success, img= cap.read()

        # find and draw poses
        img=detector.findPose(img)

        # find the position of poses landmarks
        lmList=detector.getPositions(img)
        if len(lmList):
            print(lmList[0]) # print nose landmark positions 


        # show frame per seconds (FPS)
        curr_time=time.time()
        fps=1/(curr_time-prev_time)   
        prev_time=curr_time
        cv2.putText(img, str(int(fps)), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)


        # create resizable window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        cv2.imshow('Image',img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()