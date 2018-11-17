3# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import glob
import scipy
from scipy import ndimage
from imutils.perspective import four_point_transform
from imutils import contours
import imutils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])

image = cv2.GaussianBlur(image, (5, 5), 5)
imgsize = (1280, 1024)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, imgsize)
width,height = imgsize

pts11 = np.float32([[width/12, height/2 - height/12],[width/2 + width/12, height/2 - height/12],[width/12, height-(height/12)],[width/2 + width/12,height]])
pts12 = np.float32([[0,0],[width,0],[0,height],[width,height]])
pts21 = np.float32([[width/2 - width/12, height/2 - height/12],[width - (width/12), height/2 - height/12],[width/2 - width/12, height],[width - width/12,height - (height/12)]])
pts22 = np.float32([[0,0],[width,0],[0,height],[width,height]])
pts31 = np.float32([[width/12, height/3],[width/2, height/3],[width/12, 2*height/3],[width/2,2*height/3]])
pts32 = np.float32([[0,0],[width,0],[0,height],[width,height]])
pts41 = np.float32([[width/2 - width/12, height/3],[width - (width/12), height/3],[width/2 - width/12, 2*height/3],[width - width/12,height - 2*height/3]])
pts42 = np.float32([[0,0],[width,0],[0,height],[width,height]])
M1 = cv2.getPerspectiveTransform(pts11,pts12)
M2 = cv2.getPerspectiveTransform(pts21,pts22)
M3 = cv2.getPerspectiveTransform(pts31,pts32)
M4 = cv2.getPerspectiveTransform(pts41,pts42)

imagelist = []

imagelist.append(cv2.warpPerspective(image,M1,imgsize))
imagelist.append(cv2.warpPerspective(image,M2,imgsize))
imagelist.append(cv2.warpPerspective(image,M3,imgsize))
imagelist.append(cv2.warpPerspective(image,M4,imgsize))


index=1

for image in imagelist:
    image = cv2.GaussianBlur(image, (5, 5), 5)

    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    std_per_row = np.std(image, axis=0)
    std = np.std(std_per_row, axis=0)

    th, image = cv2.threshold(image, average_color + 5 * std, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=3)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, image)

    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print("Result: ")
    print(index)
    print(text)
    index += 1

    cv2.imshow("Image", image)
    cv2.waitKey(0)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
