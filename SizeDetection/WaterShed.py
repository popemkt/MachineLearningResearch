import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils

kernel = np.ones((3, 3), np.uint8)


img = cv.imread('F:/Capstone/Playground/SizeDetection/img/10.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('Output', imutils.resize(thresh, height=1080))
# noise removal
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imshow('BG', imutils.resize(opening, height=1080))
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow('BG', imutils.resize(sure_bg, height=1080))
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
cv.imshow('FG', imutils.resize(sure_fg, height=1080))
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv.imshow("output", img)
cv.waitKey(0)
