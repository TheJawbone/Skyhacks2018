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

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (640, 480))
#
image = cv2.GaussianBlur(image, (5, 5), 5)

average_color_per_row = np.average(image, axis=0)
average_color = np.average(average_color_per_row, axis=0)
std_per_row = np.std(image, axis=0)
std = np.std(std_per_row, axis=0)

th, image=cv2.threshold(image, average_color + 5*std, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, image)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
#cv2.imshow("Output", image)
cv2.waitKey(0)