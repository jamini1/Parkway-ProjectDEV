from os.path import sep, expanduser, isdir, dirname
from kivy.config import Config

Config.set('kivy', 'exit_on_escape', 1)
Config.set('input', 'mouse', 'mouse, disable_multitouch')
# Config.set('graphics', 'fullscreen', 'auto')

import kivymd.theming

import shutil
import csv
import pandas as pd

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, StringProperty
from PIL import *
from PIL import Image
from userdatabase import Userdatabase
from placeholderOCR import countTo1000
import matplotlib.pyplot as plt
import numpy as np


# this is the login screen
class LoginWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def loginBtn(self):
        self._shadow = App.get_running_app().theme_cls.quad_shadow

        #this function validates the username and password for login feature
        if db.validate(self.email.text, self.password.text):
            MainWindow.current = self.email.text
            self.reset()
            """Loggedin()"""             #removed because annoying
            self.manager.current = "welcome"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        self.manager.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""

    def show_password(self, field, button):  # currently inoperational
        """
         Called when you press the right button in the password field
         for the screen TextFields.

         instance_field: kivy.uix.textinput.TextInput;
         instance_button: kivymd.button.MDIconButton;

         """
        # Show or hide text of password, set focus field
        # and set icon of right button.
        field.password = not field.password
        field.focus = True
        button.icon = "eye" if button.icon == "eye-off" else "eye-off"


class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count(
                "@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)

                self.reset()
                # calls the popup to confirm user is successfully created
                Submit_Success()
                self.manager.current = "login"
            else:
                invalidForm()
        else:
            invalidForm()

    def login(self):
        self.reset()
        self.manager.current = "login"

    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""


# this screen displays a welcome message and offers a PDF to be uploaded
class MainWindow(Screen):
    def changeScreen(self):
        self.manager.current = 'pdfuploaded'


# this screen will hold the loading animation
class LoadingPage(Screen):

    def on_pre_enter(self, *args):
        self.ids.loadingicon.source = "giffy.zip"

    def on_enter(self):
        # required to declare global variables
        global output
        # saving OCR function's output to a global variable in order to use later
        #output = ocr_function.ocr(ocr_function)
        value = countTo1000.placeholder_ocr(countTo1000)
        self.manager.current = 'ocrcomplete'


# this screen will show the uploaded PDF image on screen and will offer chance to
# upload a different file or convert to OCR
class PDFIsUploaded(Screen):
    def on_pre_enter(self):
        # assigns user-selected image from open_file function to the Image widget
        self.ids.image.source = image_file_directory

    #### bug here ####
    def clear_image(self):
        self.ids.image.source = ""

    ### picture does not update ###
    def assign_image(self):
        self.ids.image.source = image_file_directory


# this screen displays the converted OCR data and offers options to either edit
# data or submit the data
class OCRComplete(Screen):
    def on_enter(self):
        self.read_file()
        self.ids.iimage.source = image_file_directory

    ### this function is supposed to read the csv file onto the GUI screen and allow users to edit ###
    def read_file(self):
        file = open("Results.csv", "r")
        #csvtext = file.read()
        #csvtext = pd.read_csv(file)
        #print(file.read())
        self.ids.input.text = file.read()

        """
        ### this one doesn't work ###
        #genfromtxt reads the csv file and returns an array
        csvfile = np.genfromtxt('Results.csv', delimiter=',')
        print(csvfile)
        #fromarray creates an image from a given array. this one returns F image mode, for float #s????
        csvimage = Image.fromarray(csvfile)
        ### saving the pic as a .tiff is useless. try another method to display image ###
        csvimage.save("pic.tiff")
        # assigns the newly created image to the picture slot on screen
        self.ids.ocrpic.source = "pic.tiff"
        """

    def edit_file(self):
        # this function will open a new window to allow the user to edit the file
        pass


# this page contains an FAQ for the program
class HelpPage(Screen):
    pass


# this is the screen manager for all the screens.
class WindowManager(ScreenManager):
    pass


# displays popup for invalid username/passwords
def invalidLogin():
    pop = Popup(title='Invalid Login',
                content=Label(text='Invalid username or password.'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# displays popup for invalid form
def invalidForm():
    pop = Popup(title='Invalid Form',
                content=Label(text='Please fill in all inputs with valid information.'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# displays when user is successfully logged in
"""def Loggedin():
    pop = Popup(title='Login',
                content=Label(text='Logged in'),
                size_hint=(None, None), size=(400, 400))
    pop.open()"""


# displays when a new user is successfully created
def Submit_Success():
    pop = Popup(title='Submitted',
                content=Label(text='Successful!'),
                size_hint=(None, None), size=(400, 400))
    pop.open()


# will display when file is successfully saved
def File_Saved():
    pop = Popup(title='File Successfully Saved',
                content=Label(text='Successful!'),
                size_hint=(None, None), size=(400,400))
    pop.open()


db = Userdatabase("users.txt")
kv = Builder.load_file("design.kv")

"""
############################# OCR SECTION STARTS HERE ######################################

# -*- coding: utf-8 -*-
""
Created on Mon Jul 15 08:02:42 2019

@author: Daniel Johnson
""

############################ IMAGE SEGMENTATION PORTION ##########################################

# Import libraries
import cv2
from keras.models import load_model
import os


# process the image
def processImage(filename):
    contour_list = []
    img = cv2.imread(filename)

    # smoothing the image
    # imageblur = cv2.GaussianBlur(img, (5,5),0)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(mean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Create bounding rectangles
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 1)
        C1 = ContourHolder(x - 2, y - 2, w + 4, h + 4)
        contour_list.append(C1)

    sortContour(contour_list)
    return contour_list, img


# Sort the contours that opencv detects using their x coordinates
def sortContour(Contour_List):
    for ch in range(len(Contour_List)):

        min_index = ch
        for j in range(ch + 1, len(Contour_List)):
            if Contour_List[min_index].getX() > Contour_List[j].getX():
                min_index = j

        Contour_List[ch], Contour_List[min_index] = Contour_List[min_index], Contour_List[ch]


# Stores information regarding bounding boxes around contours
class ContourHolder(object):
    def __init__(self, xCoord, yCoord, width, height):
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.width = width
        self.height = height

    def getX(self):
        return self.xCoord

    def getY(self):
        return self.yCoord

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height


# Stores the pictures to be used for later
def createPictureList(contour_list, image):
    count = 0
    for c in contour_list:
        count += 1
        roi = image[c.getY():c.getY() + c.getHeight(), c.getX():c.getX() + c.getWidth()]
        cv2.imwrite(str(count) + '.png', roi)
    return count


# Show image with contours
### NOT BEING CALLED ###
def showPicture(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Prepare the image to be processed
# corrects size, shape, and color of each temporary PNG image file
def prepareDigit(image):
    img = cv2.imread(image)
    # digits are usually 28 pixels by 28 pixels
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, [1, 28, 28, 1])

    return img


# Predicts the numbers as listed in the cropped image file
def processContours(digits, model):
    results = []
    for number in range(1, digits + 1):
        image = prepareDigit('{0}.png'.format(number))
        ### this seems to be the ocr execution line ###
        output = model.predict(image)
        # return indices of the max element of the array in a particular axis
        results.append(np.argmax(output))

# Returns the predicted numbers
    return results


# clears all temporary image files
def cleanFolder(digits):
    for number in range(1, digits + 1):
        os.remove('{0}.png'.format(number))


########################################### Process Form ##############################################


def crop(image_path, coords, saved_location):
    ""
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    ""
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)


def getDate(image):
    crop(image, (50, 690, 340, 840), 'Date1.jpg')
    crop(image, (50, 875, 340, 1030), 'Date2.jpg')
    crop(image, (50, 1065, 340, 1220), 'Date3.jpg')
    crop(image, (50, 1250, 340, 1420), 'Date4.jpg')
    crop(image, (50, 1440, 340, 1600), 'Date5.jpg')


def getTime(image):
    # Time
    ### second argument is for coordinates--need to change ###
    crop(image, (360, 685, 640, 835), 'TIME1.jpg')
    crop(image, (360, 875, 640, 1025), 'TIME2.jpg')
    crop(image, (360, 1065, 640, 1215), 'TIME3.jpg')
    crop(image, (360, 1255, 640, 1405), 'TIME4.jpg')
    crop(image, (360, 1435, 640, 1610), 'TIME5.jpg')


def getPH(image):
    # pH
    crop(image, (660, 675, 945, 847), 'pH1.jpg')
    crop(image, (660, 865, 945, 1036), 'pH2.jpg')
    crop(image, (660, 1055, 945, 1226), 'pH3.jpg')
    crop(image, (660, 1245, 945, 1418), 'pH4.jpg')
    crop(image, (660, 1435, 945, 1600), 'pH5.jpg')


def getTempC(image):
    # tempC
    crop(image, (963, 685, 1250, 850), 'tempC1.jpg')
    crop(image, (963, 875, 1250, 1035), 'tempC2.jpg')
    crop(image, (963, 1065, 1250, 1226), 'tempC3.jpg')
    crop(image, (963, 1245, 1250, 1416), 'tempC4.jpg')
    crop(image, (963, 1445, 1250, 1605), 'tempC5.jpg')


def getAdjustedPH(image):
    # adjusted ph
    crop(image, (1270, 680, 1555, 846), 'adjustedPH1.jpg')
    crop(image, (1270, 870, 1555, 1036), 'adjustedPH2.jpg')
    crop(image, (1270, 1060, 1555, 1225), 'adjustedPH3.jpg')
    crop(image, (1270, 1250, 1555, 1415), 'adjustedPH4.jpg')
    crop(image, (1270, 1440, 1555, 1604), 'adjustedPH5.jpg')


def translatePH(total, model, text_file):
    f = open(text_file, 'a')
    f.write('PH,')
    # for loop is meant for partially naming files
    for number in range(1, total + 1):
        # creates the file called 'pH(number)'
        filename = 'pH{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        # the below commands show where the program is making contours
        #plt.imshow(img)
        #print(contour_list)
        #plt.show()
        # createPictureList creates temporary images
        count = createPictureList(contour_list, img)
        # calls actual OCR function
        resultList = processContours(count, model)
        # deletes temporary PNG files
        cleanFolder(count)

        # this says if 3 numbers are read
        ### make a test to see WHY just 3 numbers? ###
        ### if len(resultList) <= 6, then loop ###
        # if there are 3 characters, this function inserts a decimal point after the second character
        ### THIS DOESN'T ACCOUNT FOR DIGITS SUCH AS 12.3 ###
        if len(resultList) == 3:
            ### this prints a period based on ASCII table, character #46 is a period ###
            resultList[1] = chr(46)
            # takes the first, second, and third characters and saves to a variable 'numbers', then writes variable to file
            numbers = '{0}{1}{2},'.format(resultList[0], resultList[1], resultList[2])
            f.write(numbers)
        # this accounts for single digit character readings
        else:
            numbers = '{0},'.format(resultList[0])
            f.write(numbers)
    f.close()
        ### NOTE - THE FORMULAS ONLY ACCOUNT FOR SINGLE OR TRIPLE CHARACTERS--NO DOUBLE OR LONGER ###


def translateTempC(total, model, text_file):
    f = open(text_file, 'a')
    f.write('\nTempC,')
    for number in range(1, total + 1):
        filename = 'tempC{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        count = createPictureList(contour_list, img)
        resultList = processContours(count, model)
        cleanFolder(count)

    ### MIGHT NEED TO BUILD A FUNCTION FOR SINGLE DIGIT CHARACTERS, POSSIBLY ALSO NEGATIVES ###
        f.write('{0}{1},'.format(resultList[0], resultList[1]))
    f.close()


def translateAdjustedPH(total, model, text_file):
    f = open(text_file, 'a')
    f.write('\nAdjusted PH,')
    for number in range(1, total + 1):
        filename = 'adjustedPH{0}.jpg'.format(number)
        contour_list, img = processImage(filename)
        count = createPictureList(contour_list, img)
        resultList = processContours(count, model)
        cleanFolder(count)

        if (len(resultList) == 3):
            resultList[1] = chr(46)
            numbers = '{0}{1}{2},'.format(resultList[0], resultList[1], resultList[2])
            f.write(numbers)
        else:
            numbers = '{0},'.format(resultList[0])
            f.write(numbers)
    f.close()


# clears the pre-existing file to replace it with new information
def erase_file_contents(text_file):
    open(text_file, 'w').close()


########################################## Master Class #####################################################

class MasterClass:

    def processImage(filename, model, output):
        getTime(filename)
        getPH(filename)
        getTempC(filename)
        getAdjustedPH(filename)

        erase_file_contents(output)

        # translateDate(5, model, output) -- currently inoperational
        # translateTime(5, model, output) -- currently inoperational
        # 5 is the number of images the program is processing
        translatePH(5, model, output)
        translateTempC(5, model, output)
        translateAdjustedPH(5, model, output)


########################################### RUN THE PROGRAM #############################################


class ocr_function():
    def ocr(argv):
        # import keras.models, .hdf data file that is saved in a hierarchical format
        model = load_model('kerasModel4.h5')
        # saves the image file that user uploaded into function
        filename = image_file_directory
        # saves all of function's output into specific file
        output = 'Results.csv'
        MasterClass.processImage(filename, model, output)
        return output

####################################### OCR FUNCTION ENDS HERE ##########################################
"""

class FrameDesign(App):
    #theme_cls sets color scheme
    theme_cls = kivymd.theming.ThemeManager()
    title = "Frame Work"

    def build(self):
        self.theme_cls.primary_palette = 'Blue'
        #this is returning the GUI structure, the ScreenManager
        wm = WindowManager()
        return wm

    # displays popup for help button
    def helpBtn(self):
        pop = Popup(title='Need help?',
                    content=Label(
                        text='1: Create an account \n2: Sign in from login screen \n3: Upload a .jpg file from your file browser \n4: Click convert file '
                             '\n5: Save converted file into your computer' '\n Hint: On the OCR complete page, \n click on the WSSC logo to upload another file.'),
                    size_hint=(None, None), size=(400, 400))
        pop.open()

    # this function opens a file manager for the user to select a JPG file
    ### tkinter - change to Kivy ###
    def open_file(self):
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        global image_file_directory

        image_file_directory = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        root.destroy()
        return image_file_directory

    # this function opens a file manager for the user to select a JPG file
    ### tkinter - change to Kivy ###
    def save_file(self):
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        f = filedialog.asksaveasfilename(defaultextension=".csv")

        shutil.copy('Results.csv', f)

        File_Saved()

        root.destroy()

    # displays a box asking user if they want to leave application
    ### tkinter - change to kivy ###
    def exit_app(self):
        # this creates a TK canvas window. Is there a way to connect the 'root' variable
        # back into Kivy? self.manager doesn't work
        root = tk.Tk()
        canvas1 = tk.Canvas(root, width=0, height=0)
        canvas1.pack()
        root.withdraw()

        MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application?',
                                           icon='warning')
        if MsgBox == 'yes':
            root.destroy()
            quit()
        else:
            tk.messagebox.showinfo('Return', 'You will now return to the application screen.')
            root.destroy()


if __name__ == '__main__':
    FrameDesign().run()
