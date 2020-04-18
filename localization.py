#!/usr/bin/env python
# coding: utf-8

# In[47]:


from pytube import YouTube
import spacy
import srt
import en_core_web_sm
import cv2
import pytesseract 
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression

from PIL import Image
import numpy as np

COLOR_KMEANS = 12
COLOR_SLACK = 30


# In[38]:


# Timestamps from scene_detection.py
timestamps = [(61.311249334564614, 233.5666641316747),
 (285.2849969036884, 440.85707854853604),
 (483.3995780868054, 694.8608257917323)]

framestamps = [(147, 560), (684, 1057), (1159, 1666)]


# In[39]:


def get_keyframes(video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(video_filename)
    frame = 0
    while True:
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield (img, frame)
        frame += 1
        
    video_cap.release()


# In[44]:


# From https://nanonets.com/blog/deep-learning-ocr/
def getEastBoxes(img):
    height, width = img.shape[0], img.shape[1]
    net = cv2.dnn.readNet('./east_model/frozen_east_text_detection.pb')
    # Dimension should be divisible by 32, closest to 1280x720 for orignal image
    newW, newH = 1280,672
    rW = width / float(newW)
    rH = height / float(newH)
    image = cv2.resize(img, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # Get probability and bounding boxes
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
    box_res = []
    for (startX, startY, endX, endY) in boxes:
      # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        box_res.append((startX, startY, endX, endY))
        
    return box_res

## Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]
        
        min_confidence = 0.2
        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < min_confidence:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # Extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # Compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            boxes.append((startX, startY, endX, endY))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return (boxes, confidence_val)

def getMainColorsInBoxes(boxes, orig):
    colors = []
    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        #extract the region of interest
        r = orig[startY:endY, startX:endX]
#         plt.figure()
#         plt.imshow(r)
        
        Z = orig.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = COLOR_KMEANS
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for c in center:
            colors.append(c)
    return colors

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def combineBoxes(boxes):
    combined_i = []
    combinations = {}
    # Find combinations in a row of boxes
    for i in range(len(boxes)):
        (_, startY, _, endY) = boxes[i]
        start = [startY, endY]

        # already combined
        if i in combined_i:
            continue

        # check to combine
        for j in range(len(boxes)):
            if i == j:
                continue
            (_, check_startY, _, check_endY) = boxes[j]
            check = [check_startY, check_endY]
            overlap = getOverlap(start, check)
            if overlap > (endY - startY) * 0.8:
                combined_i.append(j)
                if not i in combinations:
                    combinations[i] = []
                combinations[i].append(j)

    # create a new box for the combination
    res = []
    for comb in combinations:
        startX, startY, endX, endY = boxes[comb]
        for j in combinations[comb]:
            (check_startX, check_startY, check_endX, check_endY) = boxes[j]
            startX = min(startX, check_startX)
            startY = min(startY, check_startY)
            endX = max(endX, check_endX)
            endY = max(endY, check_endY)
        startX -= 10
        endX += 10
        startY -= 10
        endY += 10
        res.append((startX, startY, endX, endY))
    return res

def readColorTextFromImage(img, color):
    b, g, r = color[0], color[1], color[2]
    
    # Show the color
#     image = np.zeros((20, 20, 3), np.uint8)
#     image[:] = (b,g,r)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     plt.imshow(image)
#     plt.show()
    
    k = COLOR_SLACK
    r_min, g_min, b_min = max(0, r-k), max(0, g-k), max(0, b-k)
    r_max, g_max, b_max = min(255, r+k), min(255, g+k), min(255, b+k)
    # Define range of color in HSV
    lower = np.array([[[b_min,g_min,r_min]]], dtype=np.uint8)
    upper = np.array([[[b_max,g_max,r_max]]], dtype=np.uint8)

    height, width = img.shape[0], img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only colors
    mask = cv2.inRange(img, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img)
#     plt.figure()
#     plt.imshow(img)
#     print(text)
    texts = text.split("\n")
    filtered_texts = []
    for t in texts:
        if t == "":
            continue
        filtered_texts.append(t)

    return filtered_texts

def readTextFromImage(img, filter_fn):
    boxes = getEastBoxes(img)
    
    if len(boxes) == 0:
        return None
    
    boxes = combineBoxes(boxes)
    colors = getMainColorsInBoxes(boxes, img)
    for c in colors:
        texts = readColorTextFromImage(img, c)
        if filter_fn(texts):
            return texts
    return None
    


# In[48]:


frame_gen = get_keyframes('data/2_Egg_Vs_95_Egg.mp4', 10)
scene_i = 0
for (img, frame) in frame_gen:

    # Start of scene
    if frame / 10 > framestamps[scene_i][0]:
        text = readTextFromImage(img, lambda x: len(x) >= 3 and "$" in x[-3:][0])
        if text == None:
            continue
        text = text[-3:]
        print('frame', frame, 'text', text)
        scene_i += 1
        if scene_i >= len(framestamps):
            break
        
    # End of scene
    if frame / 10 > framestamps[scene_i][1]:
        scene_i += 1
        if scene_i >= len(framestamps):
            break
        continue

