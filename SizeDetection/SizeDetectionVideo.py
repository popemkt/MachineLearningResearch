from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

##### Fine tuning #####

alpha = 1.0  # contrast control
beta = 0    # brightness control
threshold1 = 40  # canny control
threshold2 = 100  # canny control
kernel = np.ones((5, 5), np.uint8)  # init

cap = cv2.VideoCapture("F:/Capstone/Playground/SizeDetection/vid/1.mp4")

def run():
    while(True):
        results = []
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(enhanced, threshold1, threshold2)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        # edged = cv2.dilate(edged, None, iterations=1)
        # edged = cv2.erode(edged, None, iterations=1)
        # edged = cv2.dilate(edged, None, iterations=1)
        # edged = cv2.erode(edged, None, iterations=1)
        # edged = cv2.dilate(edged, None, iterations=1)
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        orig = image.copy()
        for c in cnts[:2]:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue
            # compute the rotated bounding box of the contour
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # # # compute the midpoint between the top-left and top-right points,
            # # # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
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
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # # if the pixels per metric has not been initialized, then
            # # compute it as the ratio of pixels to supplied metric
            # # (in this case, inches)
            # # if pixelsPerMetric is None:
            # #     pixelsPerMetric = dB / args["width"]
            # # compute the size of the object
            dimA = dA
            dimB = dB
            # # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}px".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            cv2.putText(orig, "{:.1f}px".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            # show the output image
            imh, imw, _ = image.shape
            print(str(box[0,1]) + " " + str(int(imw/2)))
            if (dimA > 300 and dimB > 100 and dimB < 130 and ((len(results) == 0 and box[0,0] > (int(imw/2) + 50)) or (len(results) == 1))):
                original = image.copy()
                # compute the rotated bounding box of the contour
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
            
        cv2.imshow('Output', imutils.resize(orig, height=600))
        cv2.waitKey(1)
        if (len(results) == 2):
            return results
run()