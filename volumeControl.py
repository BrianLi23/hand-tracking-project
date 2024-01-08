import cv2 as cv
import mediapipe as mp
import handTrackingModule as htm
import time
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

camWidth, camHeight = 1280, 720

capture = cv.VideoCapture(0)
capture.set(3, camWidth) # 3 is width
capture.set(4, camHeight) # 4 is height
prevTime = 0

detector = htm.handDetector(detectionCon=0.8)

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange() # Range is -63.5 to 0
volume.SetMasterVolumeLevel(0, None)
volumeMin = volumeRange[0]
volumeMax = volumeRange[1]


while True:
    success, img = capture.read()

    img = detector.findHands(img)
    landMarks = detector.findPosition(img, draw=False)
    if len(landMarks) != 0:

        # Points of index and thumb
        x1, y1 = landMarks[4][1], landMarks[4][2]
        x2, y2 = landMarks[8][1], landMarks[8][2]

        # Center of line
        centerX, centerY = (x1 + x2)//2, (y1+y2)//2

        # Length of line
        length = math.hypot(x2 - x1, y2 - y1)

        cv.circle(img, (x1, y1), 15, (255, 0, 0), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 0), cv.FILLED)

        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Hand Range 20 - 320 | Volume Range: -63.5 - 0
        currVolume = np.interp(length, [20, 320], [volumeMin, volumeMax])
        volume.SetMasterVolumeLevel(currVolume, None)


    # Add frame rate
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv.putText(img, f' FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow("Video", img)

    cv.waitKey(1)