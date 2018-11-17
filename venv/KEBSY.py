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
from keras.preprocessing.image import img_to_array

from keras.models import model_from_json

with open('model_architecture.json', 'r') as f:
    model_uic = model_from_json(f.read())

model_uic.load_weights('weights_model.h5')

with open('model_architecture_gaps.json', 'r') as f:
    model_gaps = model_from_json(f.read())

model_gaps.load_weights('weights_model_gaps.h5')

trainpath="Training\\0_4\\0_4_left"

traintrainlist = list()

for filename in os.listdir(trainpath):
    traintrainlist.append(os.getcwd()+"\\"+trainpath+"\\"+filename)

for file in traintrainlist:
    newimage = cv2.imread(os.getcwd()+"\\"+trainpath+"\\0_4_left_108.jpg")

    image = cv2.GaussianBlur(newimage, (5, 5), 5)
    imgsize = (1280, 1024)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, imgsize)
    width,height = imgsize

    print(file)
    pts11 = np.float32([[0, height/3], [width, height/3],
                        [width / 12, height - (height / 12)], [width -(width / 12), height - (height / 12)]])
    pts12 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts21 = np.float32([[width / 3, height/3], [width, height/3],
                        [width / 3, height], [width - width / 12, height - (height / 12)]])
    pts22 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # pts11 = np.float32([[width/12, height/2 - height/12],[width/2 + width/12, height/2 - height/12],[width/12, height-(height/12)],[width/2 + width/12,height]])
    # pts12 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # pts21 = np.float32([[width/2 - width/12, height/2 - height/12],[width, height/2 - height/12],[width/2 - width/12, height],[width - width/12,height - (height/12)]])
    # pts22 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    pts31 = np.float32([[width/12, height/3],[width/2, height/3],[width/12, 2*height/3],[width/2,2*height/3]])
    pts32 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    pts41 = np.float32([[width/2 - width/12, height/3],[width - (width/12), height/3],[width/2 - width/12, 2*height/3],[width - width/12,height - 2*height/3]])
    pts42 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    pts51 = np.float32([[width/3, height/3],[2*width/3, height/3],[width/3 + width/12, height - height/12],[2*width/3 - width/12, height - height/12]])
    pts52 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M1 = cv2.getPerspectiveTransform(pts11,pts12)
    M2 = cv2.getPerspectiveTransform(pts21,pts22)
    # M3 = cv2.getPerspectiveTransform(pts31,pts32)
    # M4 = cv2.getPerspectiveTransform(pts41,pts42)
    # M5 = cv2.getPerspectiveTransform(pts51,pts52)

    imagelist = []

    imagelist.append(cv2.warpPerspective(image,M1,imgsize))
    imagelist.append(cv2.warpPerspective(image,M2,imgsize))
    # imagelist.append(cv2.warpPerspective(image,M3,imgsize))
    # imagelist.append(cv2.warpPerspective(image,M4,imgsize))
    # imagelist.append(cv2.warpPerspective(image,M5,imgsize))
    #

    index=1

    for image in imagelist:
        image = cv2.GaussianBlur(image, (5, 5), 5)

        average_color_per_row = np.average(image, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        std_per_row = np.std(image, axis=0)
        std = np.std(std_per_row, axis=0)

        th, image = cv2.threshold(image, average_color + 5 * std, 255, cv2.THRESH_BINARY)


        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)

        thirdlinecount=8
        thirdnumber=[]
        listindex=0
        text = pytesseract.image_to_string(Image.open(filename))
        linelist = str.splitlines(text)
        if len(linelist) > 2:
            while thirdlinecount != 0 and listindex < len(linelist):
                thirdlinecount=8
                thirdnumber = []
                for character in linelist[listindex]:
                    if str.isdecimal(character) and thirdlinecount > 0:
                        thirdnumber.append(character)
                        thirdlinecount -= 1
                listindex +=1
            print(thirdnumber)
        os.remove(filename)
        print("Result: ")
        print(index)
        print(text)
        index += 1
        image = cv2.resize(image, (512, 512))
        cv2.imshow("Image", image)
        cv2.waitKey(0)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
