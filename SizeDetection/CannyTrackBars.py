import cv2
import numpy as np
import imutils


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
    cv2.imshow('canny', imutils.resize(edge, height=800))


if __name__ == '__main__':

    original = cv2.imread("F:/Capstone/Playground/SizeDetection/img/13.jpg", 1)
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
