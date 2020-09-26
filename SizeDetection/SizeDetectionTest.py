from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="path to the input image", default="F:/Capstone/Playground/SizeDetection/img/1.jpg")
ap.add_argument("-w", "--width", type=float, required=False, nargs="?",
                help="width of the left-most object in the image (in inches)", default=1.0)
args = vars(ap.parse_args())

##### Fine tuning #####

alpha = 1.0  # Simple contrast control
beta = 80    # Simple brightness control
threshold1 = 40
threshold2 = 100
minimumContourArea = 100

##### Preprocessing #####
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Gray and blurred", gray)

ret, enhanced = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("Contrasted", enhanced)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(enhanced, threshold1, threshold2)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("Edged", edged)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
pixelsPerMetric = None
orig = image.copy()
for c in cnts[:2]:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < minimumContourArea:
        continue
    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)

    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    # box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    # (tl, tr, br, bl) = box
    # (tltrX, tltrY) = midpoint(tl, tr)
    # (blbrX, blbrY) = midpoint(bl, br)
    # # # compute the midpoint between the top-left and top-right points,
    # # # followed by the midpoint between the top-righ and bottom-right
    # (tlblX, tlblY) = midpoint(tl, bl)
    # (trbrX, trbrY) = midpoint(tr, br)
    # # # draw the midpoints on the image
    # # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    # # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    # # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    # # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # # # draw lines between the midpoints
    # # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    # #          (255, 0, 255), 2)
    # # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
    # #          (255, 0, 255), 2)
    # # # compute the Euclidean distance between the midpoints
    # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # # if the pixels per metric has not been initialized, then
    # # compute it as the ratio of pixels to supplied metric
    # # (in this case, inches)
    # # if pixelsPerMetric is None:
    # #     pixelsPerMetric = dB / args["width"]
    # # compute the size of the object
    # dimA = dA
    # dimB = dB
    # # draw the object sizes on the image
    # cv2.putText(orig, "{:.1f}px".format(dimA),
    #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 2)
    # cv2.putText(orig, "{:.1f}px".format(dimB),
    #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.65, (255, 255, 255), 2)
    # show the output image
cv2.imshow('Output', imutils.resize(orig, height=1080))
cv2.waitKey(0)
