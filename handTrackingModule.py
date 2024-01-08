import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=3, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplexity = modelComplexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMs in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (244, 2, 32), cv.FILLED)

        return lmList



def main():
    prevTime = 0
    currTime = 0
    capture = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = capture.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv.putText(img, str(fps), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

        cv.imshow("Video", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()