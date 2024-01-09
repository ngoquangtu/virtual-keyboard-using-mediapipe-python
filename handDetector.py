# hand_detector.py

import cv2 as cv
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo=0, draw=True):
        x_list = []
        y_list = []
        self.lm_list = []
        bbox_info = []
        # self.results=self.hands.process(img,cv.COLOR_BGR2RGB)
        if self.results.multi_hand_landmarks!=None:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([idx, cx, cy])
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox_info = xmin, ymin, xmax, ymax
            if draw:
                cv.rectangle(img, (bbox_info[0] - 20, bbox_info[1] - 20),
                             (bbox_info[2] + 20, bbox_info[3] + 20), (0, 255, 0), 2)

        return self.lm_list, bbox_info

    def distance(self, point1, point2):
        return (point1[1] - point2[1]) * (point1[1] - point2[1]) + (point1[2] - point2[2]) * (point1[2] - point2[2])

    def get_fingers(self, img, handNo=0):
        fingers = [1, 1, 1, 1, 1]
        lm_list = self.find_position(img,handNo=handNo)
        try:
            if abs(lm_list[3][1] - lm_list[0][1]) < (lm_list[2][1] - lm_list[0][1]) or self.distance(lm_list[0], lm_list[
                2]) > self.distance(lm_list[4], lm_list[0]):
                fingers[0] = 0
            if self.distance(lm_list[0], lm_list[6]) > self.distance(lm_list[8], lm_list[0]):
                fingers[1] = 0
            if self.distance(lm_list[0], lm_list[10]) > self.distance(lm_list[0], lm_list[12]):
                fingers[2] = 0
            if self.distance(lm_list[0], lm_list[14]) > self.distance(lm_list[0], lm_list[16]):
                fingers[3] = 0
            if self.distance(lm_list[0], lm_list[18]) > self.distance(lm_list[0], lm_list[20]):
                fingers[4] = 0
        except:
            raise Exception("NO Hand Found")
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        lm_list = self.lm_list
        x1, y1 = lm_list[p1][1], lm_list[p1][2]
        x2, y2 = lm_list[p2][1], lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
