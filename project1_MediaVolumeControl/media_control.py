import time
import keyboard

class MediaController:
    def __init__(self, coolDown=0.5):
        self.coolDown = coolDown
        self.last_action = 0

    def isFingerUP(self, lmList):
        Fingers = []
        # Thumb : compare x
        Fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

        # Other fingers : compare y
        for id in [8, 12, 16, 20]:
            Fingers.append(1 if lmList[id][2] < lmList[id-2][2] else 0)

        return Fingers

    def controlMedia(self, Fingers):
        # CoolDown to prevent repeated actions
        if time.time() - self.last_action < self.coolDown:
            return

        # All fingers down → Play/Pause
        if Fingers == [0,0,0,0,0]:
            keyboard.press_and_release("play/pause media") 
            print("Play/Pause")

        self.last_action = time.time()
