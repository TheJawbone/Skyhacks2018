3# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import operator
import re
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

csv_header="team_name;train_number;left_right;frame_number;wagon;uic_0_1;uic_label\n"
submission_file=open("Nuts4DonutsSubmission.csv", "w+")
submission_file.write(csv_header)
csv_string = "Nuts4Donuts;51;left;"

frame_index = 0
wagon_number = 0
uic_bool = "locomotive"
pixel = np.array([0, 0, 0])
find_gap=0

traintrainlist = list()
fileSortDict = dict()

for justfilename in os.listdir(trainpath):
    filename = os.getcwd()+"\\"+trainpath+"\\"+justfilename
    traintrainlist.append(filename)

    if not filename[len(filename) - 7].isdecimal() and not filename[len(filename) - 6].isdecimal():
        fileSortDict[int(filename[-5:-4])] = filename
    elif not filename[len(filename) - 7].isdecimal():
        fileSortDict[int(filename[-6:-4])] = filename
    else:
        fileSortDict[int(filename[-7:-4])] = filename

standardred=0
standardgreen=0
standardblue=0

sortedList = sorted(fileSortDict.items(), key=operator.itemgetter(0))
import math
fileindex = 0
for item in sortedList:
    new_csv_string=csv_string
    new_csv_string += str(frame_index)+';'
    frame_index+=1

    file = item[1]
    newimage = cv2.imread(file)
    predimage = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
    predimage = cv2.resize(predimage, (128, 128))
    predimage = cv2.GaussianBlur(predimage, (5, 5), 5)
    predimage = cv2.Canny(predimage, 50, 250, edges=25)
    predimage = np.reshape(predimage, (1, predimage.shape[0], predimage.shape[1], 1))
    prediction=model_gaps.predict(predimage)[0] > 0.5
    if prediction[0] > 0.5 and find_gap == 1:
        print("Found gap!")
        wagon_number+=1
        find_gap = 0
    elif prediction[0] < 0.5 and find_gap == 0:
        find_gap=1
    # width,height,ch=newimage.shape
    # middlepixel = newimage[int(width/2), int(1)]
    # topleftpixel = newimage[int(width/2) -1, int(1) - 1]
    # toprightpixel = newimage[int(width/2) + 1, int(1) - 1]
    # lowerrightpixel = newimage[int(width/2) + 1, int(1) + 1]
    # lowerleftpixel = newimage[int(width/2) - 1, int(1) + 1]
    # # width, height, ch = newimage.shape
    # # middlepixel = newimage[int(width / 2), 1)]
    # # topleftpixel = newimage[int(width / 2) - 1, 0]
    # # toprightpixel = newimage[int(width / 2) + 1, 0]
    # # lowerrightpixel = newimage[int(width / 2) + 1, 2]
    # # lowerleftpixel = newimage[int(width / 2) - 1, 2]
    #
    # # width,height,ch=newimage.shape
    # # middlepixel = newimage[686, 754]
    # # topleftpixel = newimage[686 -1, 751 -1]
    # # toprightpixel = newimage[686 + 1, 751 - 1]
    # # lowerrightpixel = newimage[686 + 1, 751 + 1]
    # # lowerleftpixel = newimage[686 - 1, 751 + 1]
    #
    # averagered=middlepixel[0]/5+topleftpixel[0]/5+toprightpixel[0]/5+lowerleftpixel[0]/5+lowerrightpixel[0]/5
    # stdred = math.sqrt(pow(middlepixel[0]-averagered,2)/5+pow(topleftpixel[0]-averagered,2)/5+pow(toprightpixel[0]-averagered,2)/5+pow(lowerleftpixel[0]-averagered,2)/5+pow(lowerrightpixel[0]-averagered,2)/5)
    # averagegreen=middlepixel[1]/5+topleftpixel[1]/5+toprightpixel[1]/5+lowerleftpixel[1]/5+lowerrightpixel[1]/5
    # stdgreen = math.sqrt(pow(middlepixel[1] - averagegreen, 2) / 5 + pow(topleftpixel[1] - averagegreen, 2) / 5 + pow(
    #     toprightpixel[1] - averagegreen, 2) / 5 + pow(lowerleftpixel[1] - averagegreen, 2) / 5 + pow(
    #     lowerrightpixel[1] - averagegreen, 2) / 5)
    #
    # averageblue=middlepixel[2]/5+topleftpixel[2]/5+toprightpixel[2]/5+lowerleftpixel[2]/5+lowerrightpixel[2]/5
    # stdblue = math.sqrt(pow(middlepixel[2] - averageblue, 2) / 5 + pow(topleftpixel[2] - averageblue, 2) / 5 + pow(
    #     toprightpixel[2] - averageblue, 2) / 5 + pow(lowerleftpixel[2] - averageblue, 2) / 5 + pow(
    #     lowerrightpixel[2] - averageblue, 2) / 5)
    # averagepixel = [averagered, averagegreen, averageblue]
    # if pixel[0] == 0:
    #     if pixel[1] == 0:
    #         if pixel[2] == 0:
    #             pixel[0]=averagered
    #             pixel[1] = averagegreen
    #             pixel[2] = averageblue
    #             standardred=stdred
    #             standardgreen=stdgreen
    #             standardblue=stdblue
    # if find_gap == 0:
    #     if abs(pixel[0] - averagered) > 2*standardred:
    #         if abs(pixel[1] - averagegreen) > 2*standardgreen:
    #             if abs(pixel[2] - averageblue) > 2*standardblue:
    #                 print("lol")
    #                 find_gap = 1
    # else:
    #     if abs(pixel[0] - averagered) < 2*standardred:
    #         if abs(pixel[1] - averagegreen) < 2*standardgreen:
    #             if abs(pixel[2] - averageblue) < 2*standardblue:
    #                 print("Found gap!")
    #                 wagon_number+=1
    #                 find_gap = 0

    new_csv_string+=str(wagon_number)+';'

    image = cv2.GaussianBlur(newimage, (5, 5), 5)
    imgsize = (1280, 1024)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, imgsize)
    width,height = imgsize

    print(file)
    pts11 = np.float32(
        [[0, height / 3], [3*width / 4, height / 3], [width / 30, height - (height/30)], [3*width/4, height]])
    pts12 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts21 = np.float32([[width/5, height/3], [4*width/5, height/3],
                        [width/5 + width / 30, height - (height / 30)], [4*width/5 -(width / 30), height - (height / 30)]])
    pts22 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts31 = np.float32([[width / 4, height/3], [width, height/3],
                        [width / 4, height], [width - width / 30, height - (height / 30)]])
    pts32 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # pts41 = np.float32(
    #     [[0, height / 2], [width / 3, height / 2], [width / 30, height - (height/6) - (height / 30)], [width / 3, height - (height/6)]])
    # pts42 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])


    # pts11 = np.float32([[width/12, height/2 - height/12],[width/2 + width/12, height/2 - height/12],[width/12, height-(height/12)],[width/2 + width/12,height]])
    # pts12 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # pts21 = np.float32([[width/2 - width/12, height/2 - height/12],[width, height/2 - height/12],[width/2 - width/12, height],[width - width/12,height - (height/12)]])
    # pts22 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # pts31 = np.float32([[width/12, height/3],[width/2, height/3],[width/12, 2*height/3],[width/2,2*height/3]])
    # pts32 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    # pts41 = np.float32([[width/2 - width/12, height/3],[width - (width/12), height/3],[width/2 - width/12, 2*height/3],[width - width/12,height - 2*height/3]])
    # pts42 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    pts51 = np.float32([[width/3, height/3],[2*width/3, height/3],[width/3 + width/12, height - height/12],[2*width/3 - width/12, height - height/12]])
    pts52 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M1 = cv2.getPerspectiveTransform(pts11,pts12)
    M2 = cv2.getPerspectiveTransform(pts21,pts22)
    M3 = cv2.getPerspectiveTransform(pts31,pts32)
    #M4 = cv2.getPerspectiveTransform(pts41,pts42)
    # M5 = cv2.getPerspectiveTransform(pts51,pts52)

    imagelist = []

    imagelist.append(cv2.warpPerspective(image,M1,imgsize))
    imagelist.append(cv2.warpPerspective(image,M2,imgsize))
    imagelist.append(cv2.warpPerspective(image,M3,imgsize))
    #imagelist.append(cv2.warpPerspective(image,M4,imgsize))
    # imagelist.append(cv2.warpPerspective(image,M5,imgsize))
    #

    index=1

    outputFile = open('output.txt', 'a')
    uicFound = False

    for image in imagelist:
        if uicFound:
            break;

        image = cv2.GaussianBlur(image, (5, 5), 5)

        uic_label = '0'


        average_color_per_row = np.average(image, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        std_per_row = np.std(image, axis=0)
        std = np.std(std_per_row, axis=0)

        th, image = cv2.threshold(image, average_color + 5 * std, 255, cv2.THRESH_BINARY)


        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)

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
            #print(thirdnumber)
        os.remove(filename)
        #print("Result: ")
        #print(index)
        #print(text)

        #FORMATOWANIE TEKSTU
        for line in linelist:
            numbersCount = 0
            for char in line:
                if str.isdecimal(char):
                    numbersCount += 1
                if not str.isalnum(char):
                    line = line.replace(char, '')
            if numbersCount > 6 and len(line) == 8:
                line = line[:-1] + '-' + line[len(line) - 1]
                line = line.replace('B', '8').replace('G', '6').replace('S', '5').replace('I', '1').replace('O', '0').replace('D', 0).replace('C', '6').replace('Z','2').replace('U','0')
                print('UIC: ' + line)
                uicFound = True
                uic_bool=1
                uic_label=line
                break
            else:
                line = '0'

        index += 1
        image = cv2.resize(image, (512, 512))
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

    if wagon_number == 0:
        new_csv_string += 'locomotive;'
    elif uicFound == True:
        new_csv_string += str(1)+';'
    else:
        new_csv_string += str(0) + ';'

    new_csv_string += uic_label + '\n'

    submission_file.write(new_csv_string)

submission_file.close()
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
