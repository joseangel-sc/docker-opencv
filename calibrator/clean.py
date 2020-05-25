import cv2
import numpy as np
import time


class Cleaner:
    def __init__(self, penval_file='penval.npy', noise=800):
        self.penval = np.load(penval_file)
        self.noise = noise
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


        self.show_stack(frame, mask)

        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
        #                                        cv2.CHAIN_APPROX_SIMPLE)
        # if contours and cv2.contourArea(max(contours,
        #                                     key=cv2.contourArea)) > self.noise:
        #     # Grab the biggest contour with respect to area
        #     c = max(contours, key=cv2.contourArea)
        #
        #     # Get bounding box coordinates around that contour
        #     x, y, w, h = cv2.boundingRect(c)
        #
        #     # Draw that bounding box
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 25, 255), 2)
        #
        # cv2.imshow('image', frame)
        # # ESC key
        # k = cv2.waitKey(5) & 0xFF
        # if k == 27:
        #     return True




if __name__ == "__main__":
    capture = Cleaner()
    for x in range(10_000):
        result = capture.loop()
        if result:
            break
    cv2.destroyAllWindows()
