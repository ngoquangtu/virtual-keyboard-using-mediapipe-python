import cv2 as cv
from pynput.keyboard import Controller
from time import sleep
import math
from handDetector import HandDetector
# import cvzone

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""
keyboard = Controller()
cap = cv.VideoCapture(0)
cap.set(3, 1280)
detector = HandDetector()
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # cvzone.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),20, rt=0)
        cv.rectangle(img, button.pos, (x + w, y + h), (0, 0, 0), cv.FILLED)
        cv.putText(img, button.text, (x + 20, y + 65),
                   cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img
class Button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))
while True:
    ret, img = cap.read()
    flipped=cv.flip(img, flipCode=1)
    img=cv.resize(flipped,(1280,720))
    img = detector.find_hands(img)
    lmList, bboxInfo = detector.find_position(img)
    img = drawAll(img, buttonList)
    if len(lmList)!=0:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            x1,y1=lmList[8][1],lmList[8][2]
            x2,y2=lmList[12][1],lmList[12][2]
            if x < x1 < x + w and y < y1 < y + h:
                cv.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (122, 0, 122), -1)
                cv.putText(img, button.text, (x + 20, y + 65), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                l=math.hypot(x2-x1, y2-y1)
                print(l)

                if l < 30:
                    keyboard.press(button.text)
                    cv.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv.FILLED)
                    cv.putText(img, button.text, (x + 20, y + 65), cv.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if not key_pressed:
                        keyboard.press(button.text)
                        key_pressed = True
                    finalText += button.text
                    # sleep(0.15)
                else:
                    key_pressed=False

    cv.rectangle(img, (50, 350), (700, 450), (0, 0, 0), cv.FILLED)
    cv.putText(img, finalText, (60, 430), cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv.imshow("Hand Tracking", img)
    cv.waitKey(1)
