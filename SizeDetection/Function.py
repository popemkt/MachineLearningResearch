from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# Path

imgPath = 'F:/Capstone/Playground/SizeDetection/img/1.jpg'

##### Fine tuning #####

alpha = 1.0  # contrast control
beta = -100    # brightness control
threshold1 = 40  # canny control
threshold2 = 100  # canny control
kernel = np.ones((5, 5), np.uint8)  # init kernel


##### Preprocessing #####

image = cv2.imread(imgPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(enhanced, threshold1, threshold2)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
original = image.copy()
cnts = np.array(list(map(lambda contour: cv2.boxPoints(
    cv2.minAreaRect(contour)), cnts[:2])), dtype="int")
print(cnts)
# for c in cnts:
#     # compute the rotated bounding box of the contour
#     box=cv2.minAreaRect(c)
#     box=cv2.boxPoints(box)
#     box=np.array(box, dtype="int")
#     cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
# cv2.imshow('Output', imutils.resize(original, height=1080))
# cv2.waitKey(0)
