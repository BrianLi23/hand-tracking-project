import cv2 as cv
import time
import os

capture = cv.VideoCapture(0)
camWidth, camHeight = 1280, 720

capture.set(3, camWidth) # 3 is width
capture.set(4, camHeight) # 4 is height

while True:
    success, img = capture.read()
    cv.imshow("Video", img)
    cv.waitKey(1)