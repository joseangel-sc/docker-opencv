import cv2
import numpy as np
import time


class Cleaner:
    def __init__(self, penval_file='penval.npy'):
        self.penval = np.load(penval_file)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.kernel = np.ones((5, 5), np.uint(8))

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_range = self.penval[0]
        upper_range = self.penval[1]

        mask = cv2.inRange(hsv, lower_range, upper_range)

        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # stack all frames and show it
        stacked = np.hstack((mask_3, frame, res))
        cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            return None


if __name__ == "__main__":
    capture = Cleaner()
    for x in range(10_000):
        capture.loop()
    capture.cap.read()
    cv2.destroyAllWindows()
