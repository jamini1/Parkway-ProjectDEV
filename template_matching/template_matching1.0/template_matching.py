import cv2
import numpy as np
import imutils
from PIL import Image
from matplotlib import pyplot as plt


# step 1: call template matching on a specific column

# step 2: find where that column is located on the form, then find boundaries of the black boxes around it.
# return the dimensions of that lower-most black box into the next function.

# step 3: call the crop function for an entire column of data using the original, the template's coordinates, and the bounding box of the lower-
# most black box. Will be executed through a for loop with a counter to keep track of the file name and the y-value.

# step 4: after the entire form has been read, call the machine learning program to read through all of the file names.
# NOTE: need to figure out how to execute this kind of function as a bulk operation, since we'll be processing multiple forms!!!


# code taken from https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
def template_match(form, columnToFind):
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(columnToFind)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)

    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(form)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    """
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    """

    #print(startX, endX)
    #print(startY, endY)
    return startX, startY, endX, endY


# crops a box out - in future, will need to be in a loop
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


# runs down the column of a template's coordinates and crops each box
def crop_function(form, x0, y0, x1, y1, template_name):
    save_file_name, col_num = getFileName(template_name)
    size_chg = y1 - y0
    # establish a counter for columns
    column = 1
    # create a baseline variable to reset the save_file_name variable
    save_name = save_file_name
    # add column number to end of save_file_name
    save_file_name += '{0}'.format(column)
    # add the file ending to end of save_file_name
    save_file_name += '.jpg'

    # crop out the first box from the form
    crop(form, (x0,(y0+size_chg), x1, (y1+size_chg)), save_file_name)

    # reset save_file_name
    save_file_name = save_name

    column += 1
    # create baseline variables for coordinate calculation
    newy0, newy1 = getCoords(y0, y1, size_chg)

    for col in range(col_num):
        newy0, newy1 = getCoords(newy0, newy1, size_chg)
        save_file_name += '{0}'.format(column)
        save_file_name += '.jpg'
        crop(form, (x0, newy0, x1, newy1), save_file_name)
        save_file_name = save_name
        column += 1


# receives coordinates of item to be cropped and returns the next row's y coordinates
def getCoords(y0, y1, size_chg):
    y0 = y1
    y1 += size_chg
    return y0, y1


# returns the name and number of columns of a particular template
def getFileName(template_name):
    if template_name == "BioSolDate.jpg": # BioSolInitDate
        col_num = 7
        save_file_name = 'InitDate'
    if template_name == 'BioSolInitTime.jpg':
        col_num = 7
        save_file_name = 'InitTime'
    if template_name == 'BioSolInitpH.jpg':
        col_num = 7
        save_file_name = 'InitpH'
    if template_name == 'BioSolInitTempC.jpg':
        col_num = 7
        save_file_name = 'InitTempC'
    if template_name == 'BioSolInitAdjpH.jpg':
        col_num = 7
        save_file_name = 'InitAdjpH'
    if template_name == 'BioSolInitInitials.jpg':
        col_num = 7
        save_file_name = 'InitInitials'

    if template_name == 'BioSol2HrTime.jpg':
        col_num = 7
        save_file_name = '2HrTime'
    if template_name == 'BioSol2HrpH.jpg':
        col_num = 7
        save_file_name = '2HrpH'
    if template_name == 'BioSol2HrTempC.jpg':
        col_num = 7
        save_file_name = '2HrTempC'
    if template_name == 'BioSol2HrAdjpH.jpg':
        col_num = 7
        save_file_name = '2HrAdjpH'
    if template_name == 'BioSol2HrInitials.jpg':
        col_num = 7
        save_file_name = '2HrInitials'

    if template_name == 'BioSol24HrTime.jpg':
        col_num = 7
        save_file_name = '24HrTime'
    if template_name == 'BioSol24HrpH.jpg':
        col_num = 7
        save_file_name = '24HrpH'
    if template_name == 'BioSol24HrTempC.jpg':
        col_num = 7
        save_file_name = '24HrTempC'
    if template_name == 'BioSol24HrAdjpH.jpg':
        col_num = 7
        save_file_name = '24HrAdjpH'
    if template_name == 'BioSol24HrInitials.jpg':
        col_num = 7
        save_file_name = '24HrInitials'
    if template_name == 'BioSolTrailer.jpg':
        col_num = 7
        save_file_name = 'trailer'

    if template_name == 'BioSolLimeBuffer7-1.jpg':
        col_num = 1
        save_file_name = 'LimeBuffer71'
    if template_name == 'BioSolLimeBuffer7-2.jpg':
        col_num = 0
        save_file_name = 'LimeBuffer72'
    if template_name == 'BioSolLimeBuffer10-1.jpg':
        col_num = 1
        save_file_name = 'LimeBuffer101'
    if template_name == 'BioSolLimeBuffer10-2.jpg':
        col_num = 0
        save_file_name = 'LimeBuffer102'
    if template_name == 'BioSolLimeBuffer1245-1.jpg':
        col_num = 1
        save_file_name = 'LimeBuffer12451'
    if template_name == 'BioSolLimeBuffer1245-2.jpg':
        col_num = 0
        save_file_name = 'LimeBuffer12452'
    if template_name == 'BioSolLimeBufferTempC-1.jpg':
        col_num = 1
        save_file_name = 'LimeTempC1'
    if template_name == 'BioSolLimeBufferTempC-2.jpg':
        col_num = 0
        save_file_name = 'LimeTempC2'
    if template_name == 'BioSolLimeDate1.jpg':
        col_num = 1
        save_file_name = 'LimeDate1'
    if template_name == 'BioSolLimeDate2.jpg':
        col_num = 0
        save_file_name = 'LimeDate2'
    if template_name == 'BioSolLimeInitials1.jpg':
        col_num = 1
        save_file_name = 'LimeInitials1'
    if template_name == 'BioSolLimeInitials2.jpg':
        col_num = 0
        save_file_name = 'LimeInitials2'
    if template_name == 'BioSolLimeTime1.jpg':
        col_num = 1
        save_file_name = 'LimeTime1'
    if template_name == 'BioSolLimeTime2.jpg':
        col_num = 0
        save_file_name = 'LimeTime2'

    return save_file_name, col_num


# if user chooses the Parkway Biosolids Log, this .py file will be called to begin.
# In the end, each form will have their own.py file

# this is hardcoded in for now, but in the real program, 'form' is a global variable
form = '3261.jpg'

x0, y0, x1, y1 = template_match(form, 'BioSolDate.jpg') #BioSolInitDate
crop_function(form, x0, y0, x1, y1, "BioSolDate.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitTime.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolInitTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolInitpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitTempC.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolInitTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolInitAdjpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolInitAdjpH.jpg")

# works for 3171_001-1, but not 3175_001
x0, y0, x1, y1 = template_match(form, 'BioSolInitInitials.jpg')
crop_function(form, x0, y0, x1, y1, 'BioSolInitInitials.jpg')


x0, y0, x1, y1 = template_match(form, 'BioSol2HrTime.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol2HrTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrTempC.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol2HrTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrAdjpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol2HrAdjpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol2HrInitials.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol2HrInitials.jpg")

# works for 3171_001-1, but not 3175_001
x0, y0, x1, y1 = template_match(form, 'BioSol2HrpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol2HrpH.jpg")


x0, y0, x1, y1 = template_match(form, 'BioSol24HrTime.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol24HrTime.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol24HrpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrTempC.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol24HrTempC.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrAdjpH.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol24HrAdjpH.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSol24HrInitials.jpg')
crop_function(form, x0, y0, x1, y1, "BioSol24HrInitials.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolTrailer.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolTrailer.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer7-1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer7-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer10-2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer10-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer1245-1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer1245-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer1245-2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer1245-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBufferTempC-1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBufferTempC-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBufferTempC-2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBufferTempC-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeDate1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeDate1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeInitials1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeInitials1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeInitials2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeInitials2.jpg")

x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer7-2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer7-2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeBuffer10-1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeBuffer10-1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeDate2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeDate2.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeTime1.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeTime1.jpg")
x0, y0, x1, y1 = template_match(form, 'BioSolLimeTime2.jpg')
crop_function(form, x0, y0, x1, y1, "BioSolLimeTime2.jpg")

""""
# Step 2: call the function to process the form. Will need to call for each template--that's 20(ish) sets of coordinates!
### in the future, this will be a branching if statement depending on which form the user selects ###

# Step 3: Using the coordinates, chop up the images from the document until bottom of black box has been reached.

# Step 4: Individual image processing? (color touch ups, etc.)

# Step 5: Read images & write into csv (formatted depending on which form??? is that possible???)

# Step 6: Return csv
"""