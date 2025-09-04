import cv2
import sys
camid = int(sys.argv[1])
cap = cv2.VideoCapture(camid)

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)