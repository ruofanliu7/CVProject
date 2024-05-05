import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                smooth_landmarks=self.smooth,
                                min_detection_confidence=self.detectionCon,
                                min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        print(cTime, self.pTime)
        fbs = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fbs)), (70, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        # some time this angle comes zero, so below conditon we added
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 1)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 1)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 1)
            # cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 2)
        return angle

points = [23,24,25,26,27,28,29,30,31,32]


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=2,
                      blockSize=4)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it


def main():

    detector = poseDetector()
    cap = cv2.VideoCapture("Juggling3.MOV")
    dir = os.listdir("pose2")

    LegLog = np.zeros((1, 4))
    frame = -1
    log = open("legLog.txt", 'w')

    while True:


        frame += 1
        img = dir[frame]
        img = cv2.imread("pose2/" + img)

        #success, img = cap.read()

        img2 = img.copy()

        img = detector.findPose(img2)
        lmList = detector.getPosition(img2)

        # x,y position of the legs in indexes 1 & 2
        # id of feature in index 0
        # 23-left hip 24-right-hip
        # 25-left knee 26-right-knee 27-left ankle
        # 28-right ankle 29-left heel 30-right heel
        # 31-left-foot-index 32-right-foot-index

        LegPos = np.zeros((1,4),  dtype=np.int8)
        for i in lmList:
            print(i)
            if i[0] in points:

                LegPos = np.array([frame, i[0],i[1],i[2]],  dtype=np.int8)

                np.vstack((LegLog, LegPos))
                LegPos = str(str(frame) + "," + str(i[0]) + "," + str(i[1]) + "," + str(i[2]))
                log.write(str(LegPos) + "\n")

                img2 = cv2.circle(img2, (i[1],i[2]), 10, (0,0,255), cv2.FILLED)
                img2 = cv2.circle(img2, (i[1], i[2]), 15, (0, 0, 255), 1)
                img2 = cv2.putText(img2, str(i[0]), (i[1] - 20, i[2] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255)
                                   , 2)
                img2 = cv2.putText(img2, str(frame), (220,50), cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 0, 255)
                                   , 2)

        cv2.imwrite("poseNon2/" + str(frame) + ".jpg", img2)

        print(LegLog)

        detector.showFps(img2)
        cv2.imshow("Image", img2)
        cv2.waitKey(1)






if __name__ == "__main__":
    main()