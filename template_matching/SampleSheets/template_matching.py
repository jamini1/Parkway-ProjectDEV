import cv2
import numpy as np
import imutils
from PIL import Image
from matplotlib import pyplot as plt


# code taken from https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
# parameters are: form (the form to be read), columnToFind (individual column headers), and
# columnHead (the entire row of column headers)
def template_match(form, columnToFind, columnHead):
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(columnHead)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Header Template", template)

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

    file_name, col_num = getFileName(columnToFind)
    # NOTE: should maybe turn getFileName into 2 functions, one to return
    # the filename and one to get # of columns
    size_change = endY - startY
    new_endy = endY + (size_change * (col_num+1))

    #print(startY, new_endy, startX, endX)
    roi = image[startY:new_endy, startX:endX]
    cv2.imwrite("chopped.jpg", roi)

    # finds the template inside the form
    template = cv2.imread(columnToFind)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Template", template)

    image = cv2.imread("chopped.jpg")
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
    """cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    img = cv2.resize(image, (960, 540))
    cv2.imshow("Image", img)
    cv2.waitKey(0)"""

    return startX, startY, endX, endY


# crops a box out
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
    crop(form, (x0, (y0 + size_chg), x1, (y1 + size_chg)), save_file_name)

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
    if template_name == "pssdate.jpg":
        col_num = 7
        save_file_name = 'ssdate'
    if template_name == 'psssvirb1.jpg':
        col_num = 7
        save_file_name = 'svirb1'
    if template_name == 'psssvirb2.jpg':
        col_num = 7
        save_file_name = 'svirb2'
    if template_name == 'pssmcrtrb1.jpg':
        col_num = 7
        save_file_name = 'mcrtrb1'
    if template_name == 'pssmcrtrb2.jpg':
        col_num = 7
        save_file_name = 'mcrtrb2'
    if template_name == 'pssrb1p4pH.jpg':
        col_num = 7
        save_file_name = 'rb1p4pH'

    if template_name == 'pssrb1p4mlss.jpg':
        col_num = 7
        save_file_name = 'rb1p4mlss'
    if template_name == 'pssrb1p4min.jpg':
        col_num = 7
        save_file_name = 'rb1p4min'
    if template_name == 'pssrb2p4pH.jpg':
        col_num = 7
        save_file_name = 'rb2p4pH'
    if template_name == 'pssrb2p4mlss.jpg':
        col_num = 7
        save_file_name = 'rb2p4mlss'
    if template_name == 'pssrb2p4min.jpg':
        col_num = 7
        save_file_name = 'rb2p4min'

    if template_name == 'pssrasmlss.jpg':
        col_num = 7
        save_file_name = 'rasmlss'
    if template_name == 'pssrb1p2DO.jpg':
        col_num = 7
        save_file_name = 'rb1p2DO'
    if template_name == 'pssrb1p3.jpg':
        col_num = 7
        save_file_name = 'rb1p3'
    if template_name == 'pssrb2p2DO.jpg':
        col_num = 7
        save_file_name = 'rb2p2DO'
    if template_name == 'pssrb2p3.jpg':
        col_num = 7
        save_file_name = 'rb2p3'

    if template_name == 'psslimesilo.jpg':
        col_num = 7
        save_file_name = 'limesilo'

    if template_name == 'pssBDBpolygas.jpg':
        col_num = 1
        save_file_name = 'BDBpolygas'
    if template_name == 'pssairscrubber.jpg':
        col_num = 0
        save_file_name = 'airscrubber'
    if template_name == 'psscarbontank1.jpg':
        col_num = 1
        save_file_name = 'cartank1'
    if template_name == 'psscarbontank2.jpg':
        col_num = 0
        save_file_name = 'cartank2'
    if template_name == 'psshypotank1.jpg':
        col_num = 1
        save_file_name = 'hypotank1'
    if template_name == 'psshypotank2.jpg':
        col_num = 0
        save_file_name = 'hypotank2'
    if template_name == 'pssbisulfatetotals.jpg':
        col_num = 1
        save_file_name = 'bisulfatetotals'
    if template_name == 'psscaustictank1.jpg':
        col_num = 0
        save_file_name = 'caustank1'
    if template_name == 'psscaustictank2.jpg':
        col_num = 1
        save_file_name = 'caustank2'
    if template_name == 'pssalumtank1.jpg':
        col_num = 0
        save_file_name = 'alumtank1'
    if template_name == 'pssalumtank2.jpg':
        col_num = 1
        save_file_name = 'alumtank2'
    if template_name == 'pssinfluentpH.jpg':
        col_num = 0
        save_file_name = 'influentpH'
    if template_name == 'pssinfluenttemp.jpg':
        col_num = 1
        save_file_name = 'influenttemp'
    if template_name == 'pssphosphorousPO4.jpg':
        col_num = 0
        save_file_name = 'phosphorousPO4'
    if template_name == 'psscarbontank1.jpg':
            col_num = 1
            save_file_name = 'cartank1'
    if template_name == 'psscarbontank2.jpg':
            col_num = 0
            save_file_name = 'cartank2'
    if template_name == 'psshypotank1.jpg':
            col_num = 1
            save_file_name = 'hypotank1'
    if template_name == 'psshypotank2.jpg':
            col_num = 0
            save_file_name = 'hypotank2'
    if template_name == 'pssbisulfatetotals.jpg':
            col_num = 1
            save_file_name = 'bisulfatetotals'
    if template_name == 'psscaustictank1.jpg':
            col_num = 0
            save_file_name = 'caustank1'
    if template_name == 'psscaustictank2.jpg':
            col_num = 1
            save_file_name = 'caustank2'
    if template_name == 'pssalumtank1.jpg':
            col_num = 0
            save_file_name = 'alumtank1'
    if template_name == 'pssalumtank2.jpg':
            col_num = 1
            save_file_name = 'alumtank2'
    if template_name == 'pssinfluentpH.jpg':
            col_num = 0
            save_file_name = 'influentpH'
    if template_name == 'pssinfluenttemp.jpg':
            col_num = 1
            save_file_name = 'influenttemp'
    if template_name == 'pssphosphorousPO4.jpg':
            col_num = 0
            save_file_name = 'phosphorousPO4'

    if template_name == 'pssammonianitro.jpg':
            col_num = 1
            save_file_name = 'ammnitro'
    if template_name == 'pssCCDO.jpg':
            col_num = 0
            save_file_name = 'CCDO'
    if template_name == 'pssCCpH.jpg':
            col_num = 1
            save_file_name = 'CCpH'
    if template_name == 'psstotalCL2.jpg':
            col_num = 0
            save_file_name = 'totalCL2'
    if template_name == 'pssCL2dose.jpg':
            col_num = 1
            save_file_name = 'CL2dose'
    if template_name == 'pssbisulfitedose.jpg':
            col_num = 0
            save_file_name = 'bisulfitedose'
    if template_name == 'pssflow.jpg':
            col_num = 1
            save_file_name = 'flow'
    if template_name == 'pssinit.jpg':
            col_num = 0
            save_file_name = 'initials'
    if template_name == 'pssAVinfluentpH.jpg':
            col_num = 1
            save_file_name = 'avinflupH'
    if template_name == 'pssAVinfluenttemp.jpg':
            col_num = 0
            save_file_name = 'avinflutemp'
    if template_name == 'pssAVCCammnitro.jpg':
            col_num = 1
            save_file_name = 'avCCammnitro'
    if template_name == 'pssAVCCDO.jpg':
            col_num = 0
            save_file_name = 'avCCDO'
    if template_name == 'pssAVCCpH.jpg':
            col_num = 0
            save_file_name = 'avCCpH'
    if template_name == 'pssAVCCtotalCL2.jpg':
            col_num = 1
            save_file_name = 'avCCtotCL2'
    if template_name == 'pssemplinit.jpg':
            col_num = 0
            save_file_name = 'emplinit'

    return save_file_name, col_num


# if user chooses the Parkway Biosolids Log, this .py file will be called to begin.
# In the end, each form will have their own.py file

# this is hardcoded in for now, but in the real program, 'form' is a global variable
form = '4003.jpg'

x0, y0, x1, y1 = template_match(form, 'pssdate.jpg', 'ssHeaderCol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssdate.jpg")
x0, y0, x1, y1 = template_match(form, 'psssvirb1.jpg', 'ssHeaderCol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psssvirb1.jpg")
x0, y0, x1, y1 = template_match(form, 'psssvirb2.jpg', 'ssHeaderCol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psssvirb2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssmcrtrb1.jpg', 'ssHeaderCol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssmcrtrb1.jpg")
x0, y0, x1, y1 = template_match(form, 'pssmcrtrb2.jpg', 'ssHeaderCol1.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssmcrtrb2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb1p4pH.jpg', 'ssHeaderCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, 'pssrb1p4pH.jpg')

x0, y0, x1, y1 = template_match(form, 'pssrb1p4mlss.jpg', 'ssHeaderCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb1p4mlss.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb1p4min.jpg', 'ssHeaderCol1-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb1p4min.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb2p4mlss.jpg', 'ssHeaderCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb2p4mlss.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb2p4pH.jpg', 'ssHeaderCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb2p4pH.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb2p4min.jpg', 'ssHeaderCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb2p4min.jpg")

x0, y0, x1, y1 = template_match(form, 'pssrasmlss.jpg', 'ssHeaderCol1-3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrasmlss.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb1p2DO.jpg', 'ssHeaderCol1-4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb1p2DO.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb1p3.jpg', 'ssHeaderCol1-4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb1p3.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb2p2DO.jpg', 'ssHeaderCol1-4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb2p2DO.jpg")
x0, y0, x1, y1 = template_match(form, 'pssrb2p3.jpg', 'ssHeaderCol1-4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssrb2p3.jpg")

x0, y0, x1, y1 = template_match(form, 'psslimesilo.jpg', 'ssHeaderCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psslimesilo.jpg")
x0, y0, x1, y1 = template_match(form, 'pssBDBpolygas.jpg', 'ssHeaderCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssBDBpolygas.jpg")
x0, y0, x1, y1 = template_match(form, 'pssairscrubber.jpg', 'ssHeaderCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssairscrubber.jpg")
x0, y0, x1, y1 = template_match(form, 'psscarbontank1.jpg', 'ssHeaderCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psscarbontank1.jpg")

x0, y0, x1, y1 = template_match(form, 'psscarbontank2.jpg', 'ssHeaderCol2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psscarbontank2.jpg")
x0, y0, x1, y1 = template_match(form, 'psshypotank1.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psshypotank1.jpg")
x0, y0, x1, y1 = template_match(form, 'psshypotank2.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psshypotank2.jpg")

x0, y0, x1, y1 = template_match(form, 'pssbisulfatetotals.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssbisulfatetotals.jpg")
x0, y0, x1, y1 = template_match(form, 'psscaustictank1.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psscaustictank1.jpg")
x0, y0, x1, y1 = template_match(form, 'psscaustictank2.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psscaustictank2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssalumtank1.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssalumtank1.jpg")

x0, y0, x1, y1 = template_match(form, 'pssalumtank2.jpg', 'ssHeaderCol2-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssalumtank2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssinfluentpH.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssinfluentpH.jpg")
x0, y0, x1, y1 = template_match(form, 'pssinfluenttemp.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssinfluenttemp.jpg")

x0, y0, x1, y1 = template_match(form, 'pssphosphorousPO4.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssphosphorousPO4.jpg")
x0, y0, x1, y1 = template_match(form, 'pssammonianitro.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssammonianitro.jpg")
x0, y0, x1, y1 = template_match(form, 'pssCCDO.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssCCDO.jpg")
x0, y0, x1, y1 = template_match(form, 'pssCCpH.jpg', 'ssHeaderCol3.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssCCpH.jpg")

x0, y0, x1, y1 = template_match(form, 'psstotalCL2.jpg', 'ssHeaderCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "psstotalCL2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssCL2dose.jpg', 'ssHeaderCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssCL2dose.jpg")
x0, y0, x1, y1 = template_match(form, 'pssbisulfitedose.jpg', 'ssHeaderCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssbisulfitedose.jpg")

x0, y0, x1, y1 = template_match(form, 'pssflow.jpg', 'ssHeaderCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssflow.jpg")
x0, y0, x1, y1 = template_match(form, 'pssinit.jpg', 'ssHeaderCol3-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssinit.jpg")
x0, y0, x1, y1 = template_match(form, 'pssAVinfluentpH.jpg', 'ssHeaderCol4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVinfluentpH.jpg")
x0, y0, x1, y1 = template_match(form, 'pssAVinfluenttemp.jpg', 'ssHeaderCol4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVinfluenttemp.jpg")

x0, y0, x1, y1 = template_match(form, 'pssAVCCammnitro.jpg', 'ssHeaderCol4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVCCammnitro.jpg")
x0, y0, x1, y1 = template_match(form, 'pssAVCCDO.jpg', 'ssHeaderCol4.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVCCDO.jpg")
x0, y0, x1, y1 = template_match(form, 'pssAVCCpH.jpg', 'ssHeaderCol4-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVCCpH.jpg")

x0, y0, x1, y1 = template_match(form, 'pssAVCCtotalCL2.jpg', 'ssHeaderCol4-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssAVCCtotalCL2.jpg")
x0, y0, x1, y1 = template_match(form, 'pssemplinit.jpg', 'ssHeaderCol4-2.jpg')
crop_function("chopped.jpg", x0, y0, x1, y1, "pssemplinit.jpg")
