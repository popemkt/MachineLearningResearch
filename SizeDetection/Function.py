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

imgPath = 'F:/Capstone/Playground/SizeDetection/img/11.jpg'

##### Fine tuning #####

alpha = 1.0  # contrast control
beta = 0    # brightness control
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
# cnts = np.array(list(map(lambda contour: cv2.boxPoints(
#     cv2.minAreaRect(contour)), cnts[:2])), dtype="int")
print(len(cnts))

results = []
for c in cnts[:2]:
    original = image.copy()
    # compute the rotated bounding box of the contour
    rect = cv2.minAreaRect(c)
    print(rect)
    box = cv2.boxPoints(rect)
    box = perspective.order_points(box)
    box = np.int0(box)
    # get width and height of the detected rectangle
    width = int(min(rect[1]))
    height = int(max(rect[1]))
    print(width)
    print(height)
    src_pts = box.astype("float32")
    # coordinate of the points in box points after rectangle has been straighttened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directy warp the rotated rectangle to get the straightened reactang;e
    warped = cv2.warpPerspective(original, M, (width, height))
    results.append(warped)
    # cv2.drawContours(original, [box.astype("int")], -1, (0, 255, 0), 2)


# cv2.imshow('Output', imutils.resize(original, height=650))
for i in range(len(results)):
    cv2.imshow(str(i), results[i])

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
