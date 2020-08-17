# import the necessary packages
import numpy as np
import time
import cv2
import os
import sys
import imutils


from array import *
from PIL import Image
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist


def load_image_into_numpy_array(image2):
  #cv2.imshow("Image2",image2)
  #print(image2.size)
  #cv2.waitKey(0)
	
  (im_width, im_height) = image2.size
	
  return np.array(image2.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def measure_roi(roi,cThr=[100,100]):
  try:
    imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
  
    kernal = np.ones((2,2))
    imgDil = cv2.dilate(imgCanny, kernal, iterations=1)
  #edged2 = cv2.erode(edged, kernal, iterations=1)
  #filtered = cv2.GaussianBlur(result, (7, 7), 0)

    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    orig = roi.copy()
    pixelsPerMetric = None
  
  #for cnt in contours:
        #area = cv2.contourArea(cnt)
        #areaMin = cv2.getTrackbarPos("Area", "Parameters")
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
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
      for (x, y) in box:
          print(x, y)
          cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
      (tl, tr, br, bl) = box

      (tltrX, tltrY) = midpoint(tl, tr)
      (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
      (tlblX, tlblY) = midpoint(tl, bl)
      (trbrX, trbrY) = midpoint(tr, br)
      #   # draw the midpoints on the image
      # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
      # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
      # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
      # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
      #   # draw lines between the midpoints
      # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
      #            (255, 0, 255), 2)
      # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
      #            (255, 0, 255), 2)
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
      print(dimA)
          #   # draw the object sizes on the image
          # cv2.putText(orig, "{:.1f}mm".format(dimA),
          #               (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
          #               0.65, (255, 255, 255), 2)
          # cv2.putText(orig, "{:.1f}mm".format(dimB),
          #               (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
          #               0.65, (255, 255, 255), 2)

          # cv2.imshow("Image"+ str(i), orig)
  except Exception as e:
    print(e)


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

# load our input image and grab its spatial dimensions
image = cv2.imread("./images/opencv_frame_0.jpg")
image2 = Image.open("./images/opencv_frame_0.jpg")
(H, W) = image.shape[:2]
print(H)
print(W)
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
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
Acount=0
Rcount=0
# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		print("i = ", i)
		detected_class = classIDs[i]
		#print(detected_class)
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
			#print(text)
			#if text == Accept:
			
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

			image_np1 = load_image_into_numpy_array(image2)
			

			roi = image_np1[y:y+h,x:x+w]
			measure_roi(roi)

		if detected_class == 1:
			Rcount = Rcount +1
                        # extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# print(y,x,h,w)
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}".format(LABELS[classIDs[i]])
			#print(text)
			#if text == Accept:
			
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

print(Acount)
print(Rcount)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
