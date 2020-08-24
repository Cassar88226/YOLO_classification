import numpy as np
import time
import cv2
import os
import sys
import imutils
import math


from array import *
from PIL import Image
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist



Title = "Identify Gherkins"

import sys, time, threading, cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
try:
    import Queue as Queue
except:
    import queue as Queue

IMG_SIZE    = 640,480          # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 1                # Scaling factor for display image
DISP_MSEC   = 30                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = False              # Flag to indicate capturing

configPath = "./yolo_custom/yolov3-custom.cfg"
weightsPath = "./yolo_custom/yolov3-custom_last.weights"
labelsPath = "./yolo_custom/obj.names"

# load the COCO class labels our YOLO model was trained on

LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


# load our YOLO object detector trained on Custom dataset (2 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def display_error_message(content):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(content)
    msg.setWindowTitle("Error")
    msg.exec_()


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def measure_roi(roi,cThr=[100,100]):
    grade1_count = 0
    grade2_count = 0
    grade3_count = 0

    try:
        imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
        imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
        imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    
        kernal = np.ones((2,2))
        imgDil = cv2.dilate(imgCanny, kernal, iterations=1)

        contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        orig = roi.copy()
        pixelsPerMetric = None
    
        if len(contours) != 0:
            cnt = max(contours, key = cv2.contourArea)

            box = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # compute the Euclidean distance between the midpoints
            dB = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dA = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, inches)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 5
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            if dimA < 1.6:
                grade1_count += 1
            elif dimA >= 1.6 and dimA < 2:
                grade2_count += 1
            else:
                grade3_count += 1
    except Exception as e:
        print(e)
    return grade1_count, grade2_count, grade3_count

# Grab images from the camera (separate thread)
def grab_images(cam_num, queue):
    # cap = cv2.VideoCapture("videos\VID5.mp4") #cam_/num-1 + CAP_API
    cap = cv2.VideoCapture(cam_num-1 + CAP_API)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
    if EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    while capturing:
        if cap.grab():
            retval, image = cap.retrieve(0)
            if image is not None and queue.qsize() < 2:
                queue.put(image)

            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MyWindow(QMainWindow):

    # Create main window
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.timer = QTimer(self)           # Timer to trigger display
        self.capture_thread = None
        self.image_path = ""
        self.from_camera = True
        self.image = None

        self.setFixedHeight(875)
        self.setFixedWidth(680)
        self.centralwidget = QWidget(self)
        if DISP_SCALE > 1:
            print("Display scale %u:1" % DISP_SCALE)

        self.disp = ImageWidget(self)
        self.disp.setGeometry(QtCore.QRect(20, 10, 640, 480))
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(279, 810, 131, 24))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_start.setFont(font)
        self.btn_start.setObjectName("btn_start")
        self.btn_capture_image = QtWidgets.QPushButton(self.centralwidget)
        self.btn_capture_image.setGeometry(QtCore.QRect(279, 840, 131, 24))
        self.btn_capture_image.setFont(font)
        self.btn_capture_image.setObjectName("btn_capture_image")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(60, 550, 591, 211))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_16 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 5, 3, 1, 1)
        self.g2_percent = QtWidgets.QLabel(self.widget)
        self.g2_percent.setObjectName("g2_percent")
        self.gridLayout.addWidget(self.g2_percent, 3, 1, 1, 1)
        self.total_bill = QtWidgets.QLabel(self.widget)
        self.total_bill.setObjectName("total_bill")
        self.gridLayout.addWidget(self.total_bill, 5, 4, 1, 1)
        self.g1_bill = QtWidgets.QLabel(self.widget)
        self.g1_bill.setObjectName("g1_bill")
        self.gridLayout.addWidget(self.g1_bill, 2, 4, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 3, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 3, 3, 1, 1)
        self.g2_bill = QtWidgets.QLabel(self.widget)
        self.g2_bill.setObjectName("g2_bill")
        self.gridLayout.addWidget(self.g2_bill, 3, 4, 1, 1)
        self.g3_percent = QtWidgets.QLabel(self.widget)
        self.g3_percent.setObjectName("g3_percent")
        self.gridLayout.addWidget(self.g3_percent, 4, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 4, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 4, 3, 1, 1)
        self.g3_bill = QtWidgets.QLabel(self.widget)
        self.g3_bill.setObjectName("g3_bill")
        self.gridLayout.addWidget(self.g3_bill, 4, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 3, 1, 1)
        self.g1_percent = QtWidgets.QLabel(self.widget)
        self.g1_percent.setObjectName("g1_percent")
        self.gridLayout.addWidget(self.g1_percent, 2, 1, 1, 1)
        self.accepted_percent = QtWidgets.QLabel(self.widget)
        self.accepted_percent.setObjectName("accepted_percent")
        self.gridLayout.addWidget(self.accepted_percent, 0, 4, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.total_counts = QtWidgets.QLabel(self.widget)
        self.total_counts.setObjectName("total_counts")
        self.gridLayout.addWidget(self.total_counts, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 3, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 1, 1, 1, 2)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 522, 71, 16))
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.image_url = QtWidgets.QLineEdit(self.centralwidget)
        self.image_url.setGeometry(QtCore.QRect(140, 522, 381, 20))
        self.image_url.setFont(font)
        self.image_url.setObjectName("image_url")
        self.btn_browse = QtWidgets.QPushButton(self.centralwidget)
        self.btn_browse.setGeometry(QtCore.QRect(530, 520, 91, 23))
        self.btn_browse.setFont(font)
        self.btn_browse.setObjectName("btn_browse")

        self.radio_image = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_image.setGeometry(QtCore.QRect(210, 780, 82, 17))
        self.radio_image.setFont(font)
        self.radio_image.setObjectName("radio_image")
        self.radio_camera = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_camera.setGeometry(QtCore.QRect(410, 780, 82, 17))
        self.radio_camera.setFont(font)
        self.radio_camera.setObjectName("radio_camera")

        self.retranslateUi()
        self.btn_start.clicked.connect(self.start)
        self.btn_browse.clicked.connect(self.browse)
        self.radio_camera.setChecked(True)
        self.radio_camera.toggled.connect(self.change_image_source)
        self.radio_image.toggled.connect(self.change_image_source)
        self.btn_capture_image.clicked.connect(self.capture_image)
        self.setCentralWidget(self.centralwidget)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        self.label_16.setText(_translate("MainWindow", "Total bill:"))
        self.g2_percent.setText(_translate("MainWindow", "0"))
        self.total_bill.setText(_translate("MainWindow", "0"))
        self.g1_bill.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "Grade 2 Percentage:"))
        self.label_11.setText(_translate("MainWindow", "Grade 2 bill:"))
        self.g2_bill.setText(_translate("MainWindow", "0"))
        self.g3_percent.setText(_translate("MainWindow", "0"))
        self.label_13.setText(_translate("MainWindow", "Grade 3 Percentage:"))
        self.label_15.setText(_translate("MainWindow", "Grade 3 bill:"))
        self.g3_bill.setText(_translate("MainWindow", "0"))
        self.label_2.setText(_translate("MainWindow", "Accepted Percentage:"))
        self.g1_percent.setText(_translate("MainWindow", "0"))
        self.accepted_percent.setText(_translate("MainWindow", "0"))
        self.label.setText(_translate("MainWindow", "Total Gherkins Count:"))
        self.total_counts.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "Grade 1 Percentage:"))
        self.label_7.setText(_translate("MainWindow", "Grade 1 bill:"))

        self.label_3.setText(_translate("MainWindow", "Image URL:"))
        self.btn_browse.setText(_translate("MainWindow", "Browse"))
        self.btn_capture_image.setText(_translate("MainWindow", "Image Capture"))
        self.radio_image.setText(_translate("MainWindow", "Image"))
        self.radio_camera.setText(_translate("MainWindow", "Camera"))


    def capture_image(self):
        global capturing
        if capturing:
            self.timer.stop()
            capturing = False

        Acount = 0
        Rcount = 0
        Grade1count = 0
        Grade2count = 0
        Grade3count = 0
        # if image source is file
        if not self.from_camera:
            # check if image path is empty
            if self.image_url.text() and os.path.isfile(self.image_path):
                self.image = cv2.imread(self.image_path)
            else:
                display_error_message("Please select image")
                return

        image = self.image
        if image is not None and len(image) > 0:

            (H, W) = image.shape[:2]
            image2 = image.copy()

            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    #print(classID)
                    confidence = scores[classID]
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.5:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)


            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes                
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    detected_class = classIDs[i]
                    if detected_class == 0:
                        Acount = Acount + 1
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # print(y,x,h,w)
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        text = "{}".format(LABELS[classIDs[i]])
                        
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                        roi = image2[y:y+h,x:x+w]
                        grade1_count, grade2_count, grade3_count = measure_roi(roi)
                        Grade1count += grade1_count
                        Grade2count += grade2_count
                        Grade3count += grade3_count

                    if detected_class == 1:
                        Rcount = Rcount +1
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = "{}".format(LABELS[classIDs[i]])
                        
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                    
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(img, self.disp, DISP_SCALE)
        
        total_count = Acount+Rcount
        self.total_counts.setText(str(Acount+Rcount))
        if total_count > 0:
            self.accepted_percent.setText(str(round(Acount*100/(Acount+Rcount), 1)))
        else:
            self.accepted_percent.setText('0')
        if Acount > 0:
            self.g1_percent.setText(str(round(Grade1count*100/Acount, 1)))
            self.g2_percent.setText(str(round(Grade2count*100/Acount, 1)))
            self.g3_percent.setText(str(round(Grade3count*100/Acount, 1)))
        else:
            self.g1_percent.setText('0')
            self.g2_percent.setText('0')
            self.g3_percent.setText('0')
        self.g1_bill.setText(str(Grade1count))
        self.g2_bill.setText(str(Grade2count))
        self.g3_bill.setText(str(Grade3count))
        self.total_bill.setText(str(Grade1count + Grade2count + Grade3count))


    def browse(self):
        if self.from_camera:
            display_error_message("Please select image mode")
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","Images (*.png *.jpg)", options=options)
        if fileName:
            self.image_path = fileName
            self.image_url.setText(fileName)
            self.image = cv2.imread(self.image_path)
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image(img, self.disp, DISP_SCALE)

    def change_image_source(self):
        global capturing
        self.from_camera = self.radio_camera.isChecked()
        if not self.from_camera:
            self.timer.stop()
        capturing = False

    # Start image capture & display
    def start(self):
        global capturing
        
        if not capturing and self.from_camera:
            capturing = True
            self.timer.timeout.connect(lambda: 
                        self.show_image(image_queue, self.disp, DISP_SCALE))
            self.timer.start(DISP_MSEC)
            self.capture_thread = threading.Thread(target=grab_images, 
                        args=(camera_num, image_queue))

            self.capture_thread.start()         # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            self.image = imageq.get()
            if self.image is not None and len(self.image) > 0:
                img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        scale = max(img.shape[1]/640, img.shape[0]/480)
        disp_size = int(img.shape[1]/scale), int(img.shape[0]/scale)
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size, 
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1], 
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)



    # Window is closing: stop video capture
    def closeEvent(self, event):
        global capturing
        capturing = False
        if self.capture_thread:
            self.capture_thread.join()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            camera_num = int(sys.argv[1])
        except:
            camera_num = 0
    if camera_num < 1:
        print("Invalid camera number '%s'" % sys.argv[1])
    else:
        app = QApplication(sys.argv)
        win = MyWindow()
        win.show()
        win.setWindowTitle(Title)
        sys.exit(app.exec_())
