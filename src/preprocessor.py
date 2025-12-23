"""
Image preprocessing, iris detection, and normalization logic.
"""
import cv2
import numpy as np
from . import config


class IrisPreprocessor:
    def __init__(self):
        self.norm_height = config.NORM_HEIGHT
        self.norm_width = config.NORM_WIDTH

    def preprocess(self, img):
        """
        Apply noise reduction and inpainting to the raw image.
        """
        _, bright_mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        img_inpainted = cv2.inpaint(img, bright_mask, 3, cv2.INPAINT_TELEA)
        img_blur = cv2.medianBlur(img_inpainted, 5)
        return img_blur

    def detect_iris_circles(self, image):
        """
        Detect pupil and iris boundaries using contours and Hough transform.
        Returns: (pupil_tuple, iris_tuple) or (None, None)
        """
        # Processing image for detection
        _, bright_mask = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
        img_inpainted = cv2.inpaint(image, bright_mask, 3, cv2.INPAINT_TELEA)
        img_blur = cv2.medianBlur(img_inpainted, 7)

        # Pupil Detection
        best_pupil = None
        best_circularity = 0
        max_pupil_area = (image.shape[0] * image.shape[1]) // 10

        for thresh in range(config.PUPIL_THRESH_START, config.PUPIL_THRESH_END, config.PUPIL_THRESH_STEP):
            _, binary = cv2.threshold(img_blur, thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < config.PUPIL_MIN_AREA or area > max_pupil_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter ** 2))

                if circularity > 0.70 and circularity > best_circularity:
                    best_circularity = circularity
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        r = int(np.sqrt(area / np.pi))
                        best_pupil = (cx, cy, r)

        # Fallback to Hough Circles for Pupil
        if best_pupil is None:
            circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
                                       param1=50, param2=15, minRadius=10, maxRadius=50)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                p = circles[0, 0]
                best_pupil = (p[0], p[1], p[2])
            else:
                return None, None

        px, py, pr = best_pupil

        # Iris Detection
        min_iris_r = int(pr * 1.5)
        max_iris_r = int(pr * 4.0)
        dx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
        edges_v = cv2.convertScaleAbs(dx)
        cv2.circle(edges_v, (px, py), int(pr * 1.2), 0, -1)
        
        circles_iris = cv2.HoughCircles(
            edges_v, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
            param1=50, param2=15,
            minRadius=min_iris_r, maxRadius=max_iris_r
        )

        best_iris = None
        if circles_iris is not None:
            circles_iris = np.float32(np.around(circles_iris))
            min_dist = float('inf')

            for c in circles_iris[0, :]:
                ix, iy, ir = c[0], c[1], c[2]
                dist = np.sqrt(float(px - ix)**2 + float(py - iy)**2)

                if dist < pr * 0.5:
                    if dist < min_dist:
                        min_dist = dist
                        best_iris = (ix, iy, ir)

        if best_iris is None:
            best_iris = (px, py, int(pr * 2.8))

        return best_pupil, best_iris

    def normalize_iris(self, img, pupil, iris):
        """
        Unwrap the circular iris region into a rectangular block (Rubber Sheet Model).
        """
        xp, yp, rp = pupil
        xi, yi, ri = iris
        polar_img = np.zeros((self.norm_height, self.norm_width), dtype=np.uint8)
        theta = np.linspace(0, 2 * np.pi, self.norm_width)

        for i in range(self.norm_width):
            t = theta[i]
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            x_p_t = xp + rp * cos_t
            y_p_t = yp + rp * sin_t
            x_i_t = xi + ri * cos_t
            y_i_t = yi + ri * sin_t

            for j in range(self.norm_height):
                r = j / self.norm_height
                x = int((1 - r) * x_p_t + r * x_i_t)
                y = int((1 - r) * y_p_t + r * y_i_t)

                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    polar_img[j, i] = img[y, x]
        return polar_img