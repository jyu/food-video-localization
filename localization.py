
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
import gc
import re
import pycountry
import os
import time

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

COLOR_KMEANS = 20
COLOR_SLACK = 30

# US states
US_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

def is_country(s):
    if len(s) <= 3:
        return False
    try:
        countries = pycountry.countries.search_fuzzy(s)
        
        if name == countries[0].name:
            return True
        
        return False
    except:
        return False

# load video names
f = open('cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()

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
        
    del image
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
    
    # No combinations found, return boxes
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
    
    del hsv, mask, res, img, pil_img
    return filtered_texts

def readTextFromImage(img, filter_fn, good_colors):
    boxes = getEastBoxes(img)
    
    if len(boxes) == 0:
        return None
#     print("found east boxes")
    
    all_texts = []
    # Try good colors
    for c in good_colors:
        texts = readColorTextFromImage(img, c)
        if filter_fn(texts):
            all_texts.append(texts)
    
    if len(all_texts) > 0:  
        print("found with good color")
        print("all texts", all_texts)
        return all_texts[0]

    # Find own good color
    boxes = combineBoxes(boxes)
#     print("total boxes", len(boxes))
    colors = getMainColorsInBoxes(boxes, img)
#     print("total colors", len(colors))

    # Optimization ideas:
    # dont check colors we've seen before
    # use colors that worked before
    
    completed_colors = [] # colors we tried but don't work
    i = 0
    for c in colors:
        # See if color already tried
        skip = False
        for completed in completed_colors:
            b_dist = abs(completed[0] - c[0])
            g_dist = abs(completed[1] - c[1])
            r_dist = abs(completed[2] - c[2])
            # Threshold for same color
            if (b_dist + g_dist + r_dist < 20):
                skip = True
                break
        if skip:
            continue
            
        texts = readColorTextFromImage(img, c)
        if filter_fn(texts):
            all_texts.append(texts)
#             good_colors.append(c)
        else:
            completed_colors.append(c)
        i += 1
    print("only ran on", i, "colors instead of", len(colors), "colors") 
            
    if len(all_texts) > 0:    
#         print("all texts", all_texts)
        return all_texts[0]
#     print("no color worked")
    return None
    

k_recall = tf.keras.metrics.Recall(thresholds=0.5)
k_precision = tf.keras.metrics.Precision(thresholds=0.5)

def f1_m(y_true, y_pred):
    precision = k_precision(y_true, y_pred)
    recall = k_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Models to find the frames with location info
loc_frame_models = []
k = 5

for i in range(k):
    model = load_model('models/text_detection_nn_' + str(i) + '.h5', {'f1_m': f1_m})
    loc_frame_models.append(model)
    
name_to_model_i = {}
model_i = 0
batches = len(names) // k
for i in range(k):
    for j in range(i * batches, (i + 1) * batches):
        name_to_model_i[names[j]] = i

# Run model to get only important frames
def getTextFramesForVideo(name):
    
    # Get index of video
    print('video index', names.index(name), 'model index', name_to_model_i[name])
    
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    found_frames = []
    model = loc_frame_models[name_to_model_i[name]]
    res = model.predict(cnn_array)
    
    for i in range(res.shape[0]):
        if res[i] > 0.5:
            found_frames.append(i * 10)
    return found_frames

def isValidLocation(texts):
    if len(texts) == 0:
        return False
    
    print('is valid input', texts)
    if len(texts) < 3:
        return False
    
    # Filter out completely non alphanumeric and not $ strs out
    
    # Make it alphanumeric
    loc_1 = re.sub(r'\W+', '', texts[-2])
    loc_2 = re.sub(r'\W+', '', texts[-1])
    if len(loc_1) <= 3 or len(loc_2) <= 3:
        return False
    
#     last_pos = texts[-1].split(' ')[-1]
#     print("state pos", state_pos)
    for pos in texts[-1].split(' '):
        if pos in US_states:
            print(pos, "is a US state, True")
            return True
        if is_country(pos):
            print(pos, "is a country, True")
            return True
    
    if "$" in texts[-3:][0]:
        print("$ in ",texts[-3:][0])
        return True

    return False

def postProcessLocation(text):
    text = ', '.join(text)
    
    out_t = ""
    for i in range(len(text)):
        t = text[i]
        if t.isalnum() or t in [' ', ',', '$']:
            out_t += t
    text = out_t
    text = text.replace(' I ', ' | ').replace(' 1 ', ' | ')
    return text

def getLocationsForVideo(name, check_transition=False):
    start = time.time()
    
    frame_gen = get_keyframes('data/'+ name + '.mp4', 10)
    scene_i = 0
    
    json_file = 'labels/' + name + '.json'
    with open(json_file) as json_file:
        data = json.load(json_file)
    framestamps = data['pred_framestamps']
    
    found_frames = getTextFramesForVideo(name)
    print("found frames", found_frames)
    
    good_colors = []      # colors that worked
    out = {
        'scene_i': [],
        'frames': [],
        'locations': [],
        'raw_locations': [],
    }
    
    transition_sub = 0
    if check_transition:
        transition_sub = 60
        
    for (img, frame) in frame_gen:
        gc.collect()
        
        end = time.time()
        # Stop if run is longer than 10 min
        if end - start > 60 * 15:
            print("more than 10 min for one scene, moving on")
            scene_i += 1
            start = time.time()
            if scene_i >= len(framestamps):
                break
                
        # Start of scene
        if frame / 10 > framestamps[scene_i][0] - transition_sub and frame in found_frames:
            print(frame, scene_i)
            text = readTextFromImage(img, isValidLocation, good_colors)
            if text == None:
                continue
            text = text[-2:]
            out['raw_locations'].append(text)
            text = postProcessLocation(text)
            print('frame', frame, 'text', text)
            out['scene_i'].append(scene_i)
            out['frames'].append(frame)
            out['locations'].append(text)
            
#             scene_i += 1
#             start = time.time()
#             if scene_i >= len(framestamps):
#                 break

        # End of scene
        if frame / 10 > framestamps[scene_i][1]:
            scene_i += 1
            if scene_i >= len(framestamps):
                break
            continue
            
    # TODO: SAVE ALL OUTPUTS FOR IMPORTANT FRAMES FOR ALL FOR OFFLINE PROCESSING
    # Save as json
    with open('pred_locations/' + name + '.json', 'w') as outfile:
        json.dump(out, outfile)

skip_list = [
    '10_Cheesesteak_Vs_120_Cheesesteak',
    '1_Sushi_Vs_133_Sushi_•_Japan', 
    '4_Burrito_Vs_32_Burrito.json',
    '350_Fish_Tacos_Vs_30_Fish_Tacos',
    '13_Korean_Soup_Vs_88_Korean_Soup',
    '11_Salad_Vs_95_Salad',
    '13_Lasagna_Vs_60_Lasagna',
    '10_Sushi_&_Burger_Vs_58_Sushi_&_Burger',
    '050_Dumpling_Vs_29_Dumplings_•_Taiwan', 
    '350_Soup_Vs_29_Soup_•_Taiwan',
    '3_Chicken_Vs_62_Chicken_•_Taiwan',
    '7_Double_Cheeseburger_Vs_25_Double_Cheeseburger',
    '10_Noodles_Vs_94_Noodles', 
    '5_Fried_Chicken_Sandwich_Vs_20_Fried_Chicken_Sandwich',
]

# Classifier to preprocess if it is a location scene or not
# Just run with white color filter to be faster
completed = os.listdir('pred_locations')
for i in range(len(names)):
    n = names[i]
    if n in skip_list or n + '.json' in completed:
        continue
    print(i, 'name', n)
    getLocationsForVideo(n)


failed_videos = [
    '1_Cookie_Vs_90_Cookie.json',
    '3_Fries_Vs_100_Fries.json',
    '29_Vs_180_Family-Style_Meats.json',
    '3_Ramen_Vs_79_Ramen_•_Japan.json',
    '1_Coffee_Vs_914_Coffee_•_Japan.json',
    '4_Burger_Vs_777_Burger.json',
    '2_Egg_Vs_95_Egg.json',
    '13_BBQ_Ribs_Vs_256_BBQ_Ribs_•_Korea.json',
    '16_Steak_Vs_150_Steak_•_Australia.json',
    '7_Pho_Vs_68_Pho.json',
    '5_Pie_Vs_250_Pie.json',
    '7_Cake_Vs_208_Cake_•_Japan.json',
    '3_Seafood_Vs_213_Seafood_•_Australia.json',
    '3_Mac_N_Cheese_Vs_195_Mac_N_Cheese.json',
]
for f in failed_videos:
    n = f.replace(".json", "")
    print(n)
    getLocationsForVideo(n)

skip_list = [
#     '350_Fish_Tacos_Vs_30_Fish_Tacos',
#     '10_Cheesesteak_Vs_120_Cheesesteak',
#     '1_Sushi_Vs_133_Sushi_•_Japan', 
#     '4_Burrito_Vs_32_Burrito',
#     '13_Korean_Soup_Vs_88_Korean_Soup',
#     '11_Salad_Vs_95_Salad',
#     '13_Lasagna_Vs_60_Lasagna',
#     '10_Sushi_&_Burger_Vs_58_Sushi_&_Burger',
#     '050_Dumpling_Vs_29_Dumplings_•_Taiwan', 
#     '350_Soup_Vs_29_Soup_•_Taiwan',
    '3_Chicken_Vs_62_Chicken_•_Taiwan',
    '7_Double_Cheeseburger_Vs_25_Double_Cheeseburger',
    '10_Noodles_Vs_94_Noodles', 
    '5_Fried_Chicken_Sandwich_Vs_20_Fried_Chicken_Sandwich',
]

for n in skip_list:
    print(n)
    getLocationsForVideo(n, check_transition=True)

# Run location post processing again
completed = os.listdir('pred_locations')
a = 0
for c in completed:
    print(c)
    with open('pred_locations/' + c) as json_file:
        data = json.load(json_file)
    res = []
    for loc in data['locations']:
        print(loc)
        loc = loc.replace(",", "").replace("| ", "").replace(" ", "+")
        print("https://www.google.com/maps/?q=" + loc)
        a += 1
    print("")
print(a)

wrong = 22
failed_videos = [
]
