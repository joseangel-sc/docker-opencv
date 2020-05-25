import sys
import time

import cv2
import numpy as np


class VideoCapture:
    def __init__(self, noise=800):
        self.cap = cv2.VideoCapture(-1)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.noise = noise
        self.kernel = np.ones((5, 5), np.uint(8))

        width = int(self.cap.get(3))
        height = int(self.cap.get(4))
        self.out = cv2.VideoWriter('canvas.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

        self.pen, self.eraser, self.background_object, self.background_threshold = VideoCapture.switcher()
        self.last_switch = time.time()
        self.erase = False

    @staticmethod
    def switcher():
        pen = cv2.resize(cv2.imread('assets/pen.png', 1), (50, 50))
        eraser = cv2.resize(cv2.imread('assets/eraser.jpg', 1), (50, 50))
        background_object = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        background_threshold = 600
        return pen, eraser, background_object, background_threshold

    def draw_or_erase(self, frame):
        top_left = frame[0:50, 0:50]
        fgmask = self.background_object.apply(top_left)
        switch_thresh = np.sum(fgmask == 255)
        if switch_thresh > self.background_threshold and (time.time() - self.last_switch) > 1:
            self.last_switch = time.time()
            if not self.erase:
                self.erase = True
            else:
                self.erase = False

    def draw_canvas(self, canvas, x1, y1, record=True):
        clear = False
        wiper_thresh = 10000
        penval = np.load('penval.npy')
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.draw_or_erase(frame)

        if canvas is None:
            canvas = np.zeros_like(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_range = penval[0]
        upper_range = penval[1]
        mask = self.create_mask(hsv, lower_range, upper_range)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > self.noise:
            co = max(contours, key=cv2.contourArea)
            x2, y2, w, h = cv2.boundingRect(co)
            area = cv2.contourArea(co)

            k = cv2.waitKey(1) & 0xFF
            if (x1 != 0 or y1 != 0) and k != ord('s'):
                if not self.erase:
                    canvas = cv2.line(canvas, (x1, y1), (x2, y2), [255, 204, 0], 8)
                else:
                    cv2.circle(canvas, (x2, y2), 50, (0, 0, 0), -1)
            x1, y1 = x2, y2
            if area > wiper_thresh:
                cv2.putText(canvas, 'Clearing Canvas', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5,
                            cv2.LINE_AA)
                clear = True
        else:
            x1, y1 = 0, 0

        frame = cv2.add(frame, canvas)
        _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(foreground, background)

        if self.erase:
            cv2.circle(frame, (x1, y1), 20, (255, 255, 255), -1)
            frame[0:50, 0:50] = self.eraser
        else:
            frame[0:50, 0:50] = self.pen
        cv2.imshow('Air draw', frame)
        if record:
            self.out.write(frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            if record:
                self.out.release()
            sys.exit(0)
        if k == ord('c') or clear:
            canvas = None
        return canvas, x1, y1

    def calibrator(self, draw_rectangle=False):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h, l_s, l_v, u_h, u_s, u_v = self.calibrate()
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        mask = self.create_mask(hsv, lower_range, upper_range)

        self.show_stack(frame, mask)
        if draw_rectangle:
            self.rectangle(mask, frame)

        return VideoCapture.safe_to_pentval(l_h, l_s, l_v, u_h, u_s, u_v)

    def create_mask(self, hsv, lower_range, upper_range):
        mask = cv2.inRange(hsv, lower_range, upper_range)
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        return mask

    def rectangle(self, mask, frame):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if contours and cv2.contourArea(max(contours,
                                            key=cv2.contourArea)) > self.noise:
            co = max(contours, key=cv2.contourArea)
            x1, y1, w, h = cv2.boundingRect(co)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 25, 255), 2)
        cv2.imshow('image', frame)

    @staticmethod
    def calibrate():
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 179, VideoCapture.nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, VideoCapture.nothing)
        cv2.createTrackbar("L - V", "Trackbars", 0, 255, VideoCapture.nothing)
        cv2.createTrackbar("U - H", "Trackbars", 179, 179, VideoCapture.nothing)
        cv2.createTrackbar("U - S", "Trackbars", 255, 255, VideoCapture.nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, VideoCapture.nothing)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        return l_h, l_s, l_v, u_h, u_s, u_v

    @staticmethod
    def nothing(*args):
        pass

    @staticmethod
    def safe_to_pentval(l_h, l_s, l_v, u_h, u_s, u_v):
        key = cv2.waitKey(1)
        if key == ord('s'):
            config = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
            print(config)
            np.save('penval', config)
            cv2.destroyAllWindows()
            return True
        return False

    @staticmethod
    def show_stack(frame, mask):
        res = cv2.bitwise_and(frame, frame, mask=mask)
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((mask_3, frame, res))
        cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))


if __name__ == "__main__":
    capture = VideoCapture()
    for _ in range(10_000):
        result = capture.calibrator(draw_rectangle=True)
        if result:
            break
    c = None
    x, y = 0, 0
    for _ in range(10_000):
        c, x, y = capture.draw_canvas(c, x, y, record=True)
    cv2.destroyAllWindows()
