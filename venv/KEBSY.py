# import the necessary packages
from keras.models import model_from_json
from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import operator

#opening CNN model
with open('model_architecture_gaps.json', 'r') as f:
    model_gaps = model_from_json(f.read())

#loading weights to CNN model
model_gaps.load_weights('weights_model_gaps.h5')

#path for test set
trainpath="Validation\\0_64_left"

#Header for csv file
csv_header="team_name,train_number,left_right,frame_number,wagon,uic_0_1,uic_label\n"
submission_file=open("Nuts4DonutsSubmission.csv", "w+")
submission_file.write(csv_header)
csv_string = "Nuts4Donuts,64,left,"

#Data for filling the csv file
frame_index = 0
wagon_number = 0
uic_bool = "locomotive"

#Variable for signalling a gap between wagons
find_gap=0

#list for accessing files in test directory
traintrainlist = list()
fileSortDict = dict()

#Sorting files in list
for justfilename in os.listdir(trainpath):
    filename = os.getcwd()+"\\"+trainpath+"\\"+justfilename
    traintrainlist.append(filename)

    numSeqActive = False
    numSeq = str()
    for index in range(len(filename) - 1, 0, -1):
        if filename[index].isdecimal():
            numSeqActive = True
            numSeq += filename[index]
        elif not filename[index].isdecimal() and numSeqActive:
            break
    numSeq = numSeq[::-1]
    fileSortDict[int(numSeq)] = filename

sortedList = sorted(fileSortDict.items(), key=operator.itemgetter(0))

#A delay in frames for finding a new gap after finding a gap
frame_delay = 15

#For loop processing files one by one for finding UIC numbers
for item in sortedList:

    #preparing a new line for csv file
    new_csv_string=csv_string
    new_csv_string += str(frame_index)+','
    frame_index+=1

    #Decrementing the delay for finding a new gap between wagons
    if frame_delay > 0:
        frame_delay -= 1

    #Reading a file
    file = item[1]
    newimage = cv2.imread(file)

    #Preprocessing the image for CNN prediction
    predimage = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
    predimage = cv2.resize(predimage, (128, 128))
    predimage = cv2.GaussianBlur(predimage, (5, 5), 5)
    predimage = cv2.Canny(predimage, 50, 250, edges=25)
    predimage = np.reshape(predimage, (1, predimage.shape[0], predimage.shape[1], 1))

    #If the image is classified as containing a gap, and the delay is 0, mark new wagon and set the delay
    prediction=model_gaps.predict(predimage)[0] > 0.5
    if prediction[0] > 0.5 and find_gap == 1 and frame_delay == 0:
        print("Found gap!")
        wagon_number+=1
        find_gap = 0
        frame_delay=15
    elif prediction[0] < 0.5 and find_gap == 0:
        find_gap=1

    #adding wagon number to csv line
    new_csv_string+=str(wagon_number)+','

    #Preprocessing image for UIC number detection
    image = cv2.GaussianBlur(newimage, (5, 5), 5)
    imgsize = (1280, 1024)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, imgsize)
    width,height = imgsize

    #Printing the file path
    print(file)

    #Setting points for perspective projection for fisheye distortion correction
    #Bottom-left region
    pts11 = np.float32(
        [[0, height / 3], [3*width / 4, height / 3], [width / 30, height - (height/30)], [3*width/4, height]])
    pts12 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    #Mid-bottom region
    pts21 = np.float32([[width/5, height/3], [4*width/5, height/3],
                        [width/5 + width / 30, height - (height / 30)], [4*width/5 -(width / 30), height - (height / 30)]])
    pts22 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    #Bottom-right region
    pts31 = np.float32([[width / 4, height/3], [width, height/3],
                        [width / 4, height], [width - width / 30, height - (height / 30)]])
    pts32 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    #Computing perspective transform for previously defined regions
    M1 = cv2.getPerspectiveTransform(pts11,pts12)
    M2 = cv2.getPerspectiveTransform(pts21,pts22)
    M3 = cv2.getPerspectiveTransform(pts31,pts32)

    #A list for storing the projected images
    imagelist = []

    #Adding perspective warps of regions to imagelist
    imagelist.append(cv2.warpPerspective(image,M1,imgsize))
    imagelist.append(cv2.warpPerspective(image,M2,imgsize))
    imagelist.append(cv2.warpPerspective(image,M3,imgsize))

    #Variable for signalling that an UIC number was detected
    uicFound = False

    #For loop for processing perspective projections of image regions
    for image in imagelist:
        #if uic nmber already found, then stop processing
        if uicFound:
            break;

        #blurring the image
        image = cv2.GaussianBlur(image, (5, 5), 5)

        #Thresholding the image using average color of image + 5 times the standard deviation
        average_color_per_row = np.average(image, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        std_per_row = np.std(image, axis=0)
        std = np.std(std_per_row, axis=0)

        th, image = cv2.threshold(image, average_color + 5 * std, 255, cv2.THRESH_BINARY)

        #Eroding and dilating the image for filling gaps in numbers
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.dilate(image, kernel, iterations=1)

        #saving a temporary image file for OCR processing
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, image)

        #Variables for processing lines read by Tesseract OCR
        thirdlinecount=8
        thirdnumber=[]
        listindex=0

        #Processing image into string by Tesseract
        text = pytesseract.image_to_string(Image.open(filename))

        #Splitting lines in processed image
        linelist = str.splitlines(text)
        os.remove(filename)

        #For loop processing lines read in image file and counting numbers
        for line in linelist:
            numbersCount = 0
            #for loop processing each character in line
            for char in line:
                #if the character is a number, increment numbersCount
                #if it's not alphanumeric, replace it with nothing
                if str.isdecimal(char):
                    numbersCount += 1
                if not str.isalnum(char):
                    line = line.replace(char, '')
            #if there were more than 6 numbers read, and the line has 8 characters,
            ## replace letters with similar looking numbers
            if numbersCount > 6 and len(line) == 8:
                line = line[:-1] + '-' + line[len(line) - 1]
                line = line.replace('B', '8')\
                    .replace('G', '6')\
                    .replace('S', '5')\
                    .replace('I', '1')\
                    .replace('O', '0')\
                    .replace('D', '0')\
                    .replace('C', '6')\
                    .replace('Z','2')\
                    .replace('U','0')
                print('UIC: ' + line)
                #signal the UIC was found and set the data for csv line
                uicFound = True

                uic_bool=1
                uic_label=line
                break
            else:
                #set line as zero if no UIC number found
                line = '0'

        #increment file index
        index += 1

    #if no gaps detected, mark wagon as locomotive
    if wagon_number == 0:
        new_csv_string += 'locomotive,'
    elif uicFound == True:
        new_csv_string += str(1)+','
    else:
        new_csv_string += str(0) + ','

    #put uic number into csv line
    new_csv_string += uic_label + '\n'

    #write line into csv file
    submission_file.write(new_csv_string)

#close the csv file
submission_file.close()
