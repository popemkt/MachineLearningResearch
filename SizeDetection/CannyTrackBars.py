from scipy.spatial import distance as dist
import cv2
import numpy as np
import imutils

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def funcCan(values=0):
    thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
    dilate = cv2.getTrackbarPos('dilate', 'canny') * 2 + 1
    erode = cv2.getTrackbarPos('erode', 'canny') * 2 + 1
    gaussian = cv2.getTrackbarPos('gaussian', 'canny') * 2 + 1

    edge = cv2.GaussianBlur(img, (gaussian, gaussian), 0)
    edge = cv2.Canny(edge, thresh1, thresh2)
    edge = cv2.dilate(edge, None if dilate == 0 else np.ones(
        (dilate, dilate), np.uint8), iterations=1)
    edge = cv2.erode(edge, None if erode == 0 else np.ones(
        (erode, erode), np.uint8), iterations=1)
    orig = img.copy()
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    for c in cnts[:2]:
        # if the contour is not sufficiently large, ignore it
        # if cv2.contourArea(c) < minimumContourArea:
        #     continue
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
    cv2.imshow('canny', imutils.resize(orig, height=800))


if __name__ == '__main__':

    original = cv2.imread("F:/Capstone/Playground/SizeDetection/img/2.jpg", 1)
    img = original.copy()
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.namedWindow('canny')

    thresh1 = 100
    thresh2 = 1
    erode = 0
    dilate = 0
    gaussian = 2
    cv2.createTrackbar('thresh1', 'canny', thresh1, 255, funcCan)
    cv2.createTrackbar('thresh2', 'canny', thresh2, 255, funcCan)
    cv2.createTrackbar('erode', 'canny', erode, 4, funcCan)
    cv2.createTrackbar('dilate', 'canny', dilate, 4, funcCan)
    cv2.createTrackbar('gaussian', 'canny', gaussian, 20, funcCan)
    funcCan(0)
    cv2.imshow('Frame', imutils.resize(original, height=800))
    cv2.waitKey(0)


cv2.destroyAllWindows()
