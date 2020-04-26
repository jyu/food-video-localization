#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pytube import YouTube
import spacy
import srt
import en_core_web_sm
import cv2
import pytesseract 
from matplotlib import pyplot as plt
from imutils.object_detection import non_max_suppression
import json
from PIL import Image
import numpy as np


# In[4]:


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


# In[16]:


name = '3_Fries_Vs_100_Fries'
frame_gen = get_keyframes('data/' + name + '.mp4', 10)
json_file = 'labels/' + name + '.json'
with open(json_file) as json_file:
    data = json.load(json_file)
framestamps = data['pred_framestamps']

print(framestamps)
show = 0
scene_i = 1
for (img, frame) in frame_gen:
    if scene_i < len(framestamps):
        frame_boundary = framestamps[scene_i][0] - 0
        
    if show > 0 or frame / 10 > frame_boundary:
        print("showing frame", frame, show)
        if scene_i < len(framestamps) and frame / 10 > frame_boundary:
            show = 60
            scene_i += 1
                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
        plt.figure()
        plt.imshow(img)
        plt.text(1500,300,"frame:\n" + str(frame))
        plt.text(1500,600,"name:\n" + str(name))
        show = max(0, show - 1)
    
    if show == 0 and frame / 10 > framestamps[1][0] + 60:
        break


# In[6]:


# aus seafood frames
# frames = [2170, 7260, 7280, 12660]

# fish taco frames
frames = [1170, 7310, 14780]
opencv_img = []
frame_gen = get_keyframes('data/' + name + '.mp4', 5)
show = 0
scene_i = 0
for (img, frame) in frame_gen:
    if frames[scene_i] == frame:
        print("showing frame", frame, show)
        cv2.imwrite("img_" + str(frame) + ".jpg", img)
        opencv_img.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img)
#         configuration = ("-l eng --oem 1 --psm 11")
#         text = pytesseract.image_to_string(pil_img, config=configuration)
#         if text == "":
#             continue
            
        plt.figure()
        plt.imshow(img)
#         plt.text(1500,150,"text:\n" + str(text))
        plt.text(1500,300,"frame:\n" + str(frame))
        scene_i += 1
        if scene_i >= len(frames):
            break


# In[8]:


# From https://nanonets.com/blog/deep-learning-ocr/
def getEastBoxes(img):
    height, width = img.shape[0], img.shape[1]
    net = cv2.dnn.readNet('./east_model/frozen_east_text_detection.pb')
    # Dimension should be divisible by 32, closest to 1280x720 for orignal image
#     newW, newH = 320, 192
#     newW, newH = 640, 352
    newW, newH = 1280,672
#     newW, newH = 1280, 736
    rW = width / float(newW)
    rH = height / float(newH)
    image = cv2.resize(img, (newW, newH))
    cv2.imwrite("east.jpg", image)
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
#             print("Got score", scoresData[i] )
            if scoresData[i] < min_confidence:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            boxes.append((startX, startY, endX, endY))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return (boxes, confidence_val)
        


# In[9]:


def detectTextInBoxes(boxes, orig):
    # initialize the list of results
    results = []
    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        #extract the region of interest
        r = orig[startY:endY, startX:endX]
        plt.figure()
        plt.imshow(r)
        #configuration setting to convert image to string.  
        configuration = ("-l eng --oem 1 --psm 7")
        ##This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)
        # append bbox coordinate and associated text to the list of results 
        results.append(((startX, startY, endX, endY), text))
    return results


# In[10]:


def getMainColorsInBoxes(boxes, orig):
    colors = []
    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        #extract the region of interest
        r = orig[startY:endY, startX:endX]
        plt.figure()
        plt.imshow(r)
        
        Z = orig.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 20
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        for c in center:
            colors.append(c)
    return colors


# In[12]:


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
    print('combinations', combinations)
    if len(combinations) == 0:
        return boxes
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


# In[13]:


def readColorTextFromImage(img, color, completed_colors):
    b, g, r = color[0], color[1], color[2]
    
    # Show the color
    image = np.zeros((20, 20, 3), np.uint8)
    image[:] = (b,g,r)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.show()
    
    k = 30
    r_min, g_min, b_min = max(0, r-k), max(0, g-k), max(0, b-k)
    r_max, g_max, b_max = min(255, r+k), min(255, g+k), min(255, b+k)
    # define range of color in HSV
    lower = np.array([[[b_min,g_min,r_min]]], dtype=np.uint8)
    upper = np.array([[[b_max,g_max,r_max]]], dtype=np.uint8)

    print('bgr', color, lower, upper)

#     lower = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
#     lower = lower[0][0]
#     upper = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
#     upper = upper[0][0]
#     print('hsv', lower, upper)
    
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
    plt.figure()
    plt.imshow(img)
    print(text)
    texts = text.split("\n")
    filtered_texts = []
    for t in texts:
        if t == "":
            continue
        filtered_texts.append(t)
        
    # Did not find any text with the color
    if len(filtered_texts) == 0:
        completed_colors.append(color)
    return filtered_texts


# In[16]:


img = opencv_img[2]
# height, width = img.shape[0], img.shape[1]
# orig = orig[height//2:height]
    
boxes = getEastBoxes(img)
print('east boxes', boxes)
for b in boxes:
    east_box = img[b[1]:b[3], b[0]:b[2]]
    plt.figure()
    plt.imshow(east_box)
boxes = combineBoxes(boxes)
print('combined boxes', boxes)
# res = detectTextInBoxes(boxes, img)
colors = getMainColorsInBoxes(boxes, img)
print('colors', colors)
print('total colors', len(colors))
completed_colors = []
i = 0
for c in colors:
    skip = False
    for completed in completed_colors:
        b_dist = abs(completed[0] - c[0])
        g_dist = abs(completed[1] - c[1])
        r_dist = abs(completed[2] - c[2])
        print(completed, c, b_dist, g_dist, r_dist)
        # Threshold for same color
        if (b_dist + g_dist + r_dist < 20):
            skip = True
            break
    if skip:
        continue
        
    i += 1
    readColorTextFromImage(img, c, completed_colors)

print(completed_colors)
print("only ran on", i, "colors instead of", len(colors), "colors")
    
# cv2.imwrite("out.jpg", orig_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
# plt.title('Output')
# plt.show()


# In[44]:


import re

def readWhiteTextFromImage(img):
    height, width = img.shape[0], img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    lower_white = np.array([230,230,230], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(img, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    img = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img)
    plt.figure()
    plt.imshow(img)
    print(text)
#     plt.text(1500,150,"text:\n" + str(text))
    texts = text.split("\n")
    filtered_texts = []
    for t in texts:
        if t == "":
            continue
        filtered_texts.append(t)

    return filtered_texts

texts = []
for img in opencv_img:
    texts.append(readWhiteTextFromImage(img))


# In[51]:


print(texts)
nlp = en_core_web_sm.load()

    
for t in texts:
    for i in t:
        doc = nlp(text)

        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

        


# In[52]:


yt = YouTube("https://www.youtube.com/watch?v=JM1Qgsh2rn0")

caption = yt.captions.get_by_language_code('en')
caption_srt = caption.generate_srt_captions()
subtitle_generator = srt.parse(caption_srt)

timestamp_to_text = {}
for i in range(len(timestamps)):
    timestamp_to_text[i] = ""
    
t_i = 0
for sub in subtitle_generator:
    start = sub.start.total_seconds()
    end = sub.end.total_seconds()
    if start > timestamps[t_i][0]:
        timestamp_to_text[t_i] += sub.content + " "
    if end > timestamps[t_i][1]:
        t_i += 1
        if t_i >= len(timestamps):
            break
        timestamp_to_text[t_i] += sub.content + " "

# print(timestamp_to_text[0])
print("NEW TEXT")
print(timestamp_to_text[1])


# In[7]:


nlp = en_core_web_sm.load()

# load video names
f = open('cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()


# In[17]:


def get_ner_loc_from_name(name):
    
    for i in range(3):
        
        f = open('transcripts/' + name + '/scene_' + str(i) + '.txt')
        lines = list(f.readlines())
        if len(lines) == 1:
            return 0
        text = ' '.join(lines[:3])
        print(text)
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "ORG":
                print(ent.text, ent.label_)
                loc = ent.text.replace(" ", "+")
                print("https://www.google.com/maps/?q=" + loc)
            
        print("")
        return 3

a = 0
for n in names:
    a += get_ner_loc_from_name(n)
print(a)

# text = """
# We're at Fay Da Bakery, and we will be showing you our curry beef puff.
# """
# doc = nlp(text)

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

# print(text)


# In[22]:


import requests

response = requests.get('https://www.google.com/maps/?q=AFRAME+CULVER+CITY+CA')
if response.history:
    print("Request was redirected")
    for resp in response.history:
        print(resp.status_code, resp.url)
    print("Final destination:")
    print(response.status_code, response.url)
else:
    print("Request was not redirected")


# In[34]:


import pycountry
countries = pycountry.countries.search_fuzzy('asdf')
print(countries[0].name)


# In[ ]:


def get_labels_from_video(name):
    f = open('pred_locations/' + name + '.json')
    data = json.load(f)
    location_frames = data['frames']
    
    # Only trian on videos with all frames labelled
    if len(location_frames) != 3:
        return
    
    frames_to_show = []
    for f in location_frames:
        print(f)
        frames_to_show.append(f - 50)
        frames_to_show.append(f)
        frames_to_show.append(f + 50)
        frames_to_show.append(f + 100)

    f = open('labels/' + name + '.json')
    data = json.load(f)
    for f in data['frame_stamps']:
        print(f)
        
    frame_gen = get_keyframes('data/' + name + '.mp4', 10)
    for (img, frame) in frame_gen:
        if frame in frames_to_show:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(img)
            plt.text(1500,300,"frame:\n" + str(frame))


for n in names[:5]:
    get_labels_from_video(n)

