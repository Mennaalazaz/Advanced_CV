import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeController:
    def __init__(self,coolDown=0.7):
        # Get default audio device and cast to IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.coolDown = coolDown
        self.last_action = 0

    def getCurrentVolume(self):
        return self.volume.GetMasterVolumeLevelScalar() # Get current volume as scalar (0.0 – 1.0)
    
    def getVolumePercent(self,vol):
        volPercent = int(vol*100) # convert to percentage
        return volPercent

    def getVolumeBar(self,volPercent):
        volBar=np.interp(volPercent,[0,100],[400,150]) # map 0–100% → bar range
        return volBar

    def setCurrentVolume(self,volPercent):
        # volume.SetMasterVolumeLevelScalar takes a float between 0→1
        self.volume.SetMasterVolumeLevelScalar(volPercent/100, None) 