import cv2
import FaceMeshModule as FMM
import time

def main():
    cap=cv2.VideoCapture('video.mp4')

    # Create face mesh detector (can track up to 3 faces)
    detector=FMM.FaceMeshDetector(maxFaces=3)
    prev_time=0

    while(True):
        success, img=cap.read()

        img,faces=detector.findFaceMesh(img)

        # Print the first face's landmarks if at least one face is detected
        if len(faces):
            print((faces[0]))

        # Resize the frame (for example, width=640, height=480)
        img = cv2.resize(img, (640, 480))

        # show fps (frame per second)
        curr_time=time.time()
        fps=1/(curr_time-prev_time)
        prev_time=curr_time
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        cv2.imshow('Image',img)
        cv2.waitKey(1)



if __name__=='__main__':
    main()