import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

capture = cv.VideoCapture(0) # 3 is virtual camera
camWidth, camHeight = 1280, 720
prevTime = 0
screenWidth, screenHeight = pyautogui.size()

# Set the video length and width
capture.set(3, camWidth) # 3 is width
capture.set(4, camHeight) # 4 is height

# Initialize detector
detector = htm.handDetector(detectionCon=0.8)

# Initialize variables for click detection
prev_tip_y = 0
click_threshold = 10  # Value determines click sensitivity

while True:
    success, img = capture.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img, draw=True)
    landMarks = detector.findPosition(img) # This is a list of all land marks

    # Position of index finger
    if len(landMarks):
        x1, y1 = landMarks[8][1], landMarks[8][2]
        real_x = (screenWidth/camWidth) * x1
        real_y = (screenHeight/camHeight) * y1
        cv.circle(img, (x1,y1), 10, (0,255, 255))

        # Check for click
        if abs(y1 - prev_tip_y) > click_threshold:
            pyautogui.click()
            cv.putText(img, "Click", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        prev_tip_y = y1

        pyautogui.moveTo(real_x, real_y, 0)




    # Steps needed, find tip of index and middle finger
    # 1. Find the landmarks of hand
    # 2. Check which fingers are up, if index only be in a
    # "moving mode" else, "clicking mode"

    cv.imshow("Video", img)
    cv.waitKey(1)