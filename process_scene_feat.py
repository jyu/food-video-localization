#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import srt
import json
import os
from pprint import pprint

import keras
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


# In[2]:


model = ResNet50(weights='imagenet')
feat_extract_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)


# In[3]:


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
    
def write_features_to_file(features, fwrite):
    for i in range(features.shape[0]):
      feat = features[i]
      line = str(feat[0])
      for m in range(1, feat.shape[0]):
          line += ';' + str(feat[m])
      line += "\n"
      fwrite.write(line)


# In[4]:


def get_cnn_features_from_video(video_filename, cnn_feat_filename, model, interval):

    frame_gen = get_keyframes(video_filename, interval)

    fwrite = open(cnn_feat_filename, 'w')
    xs = []
    processed = 0
    for (img, frame) in frame_gen:
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      im_pil = Image.fromarray(img)
      im_resized = im_pil.resize((224, 224))
      x = image.img_to_array(im_resized)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      xs.append(x[0])

      if len(xs) == 256:
        # Predict for batch of 256
        xs = np.array(xs)
        preds = model.predict(xs)
        preds = preds.reshape((preds.shape[0],-1))
        write_features_to_file(preds, fwrite)
        xs = []
        processed += 256
        print("Processed", processed * 10, "frames")

    # For last images
    xs = np.array(xs)
    if xs.shape[0] == 0:
      return
    preds = model.predict(xs)
    preds = preds.reshape((preds.shape[0],-1))
    write_features_to_file(preds, fwrite)

    fwrite.close()


# In[5]:


def get_scene_label_from_video(video_filename, label_filename, frame_stamps, interval):
    frame_gen = get_keyframes(video_filename, interval)

    fwrite = open(label_filename, 'w')
    
    scene_i = 0
    for (img, frame) in frame_gen:
        # End of the video
        if scene_i >= len(frame_stamps):
            fwrite.write("end\n")
            continue
            
        scene_end = frame_stamps[scene_i][1]
        
         # End of the scene
        if frame > scene_end:
            scene_i += 1
            print("On frame", frame, "moving on to scene", scene_i)
        
            # End of the video
            if scene_i >= len(frame_stamps):
                fwrite.write("end\n")
                continue
        
        scene = frame_stamps[scene_i]
        scene_start = scene[0]
        scene_end = scene[1]
            
        # Inside scene
        if frame >= scene_start and frame < scene_end:
            fwrite.write(scene[2] + "\n")
        else:
            fwrite.write("NULL\n")


# In[6]:


def extract_feat_for_video(name):
    print("Extracting for video", name)
    json_file = 'labels/' + name + '.json'
    json_file = open(json_file)
    data = json.load(json_file)
    frame_stamps = data['frame_stamps']
    print("Extracting CNN features")
    get_cnn_features_from_video(data['video_path'], 'cnn/' + name + '.feat', feat_extract_model, 10)
    print("Extracting labels")
    get_scene_label_from_video(data['video_path'], 'scene_labels/' + name + '.labels', frame_stamps, 10)
    
    print("Checking dimensions")
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_array = np.genfromtxt('scene_labels/' + name + '.labels', delimiter=";")
    print("cnn array dim", cnn_array.shape, "scene array dim", scene_array.shape)
    assert(cnn_array.shape[1] == 2048)
    assert(scene_array.shape[0] == cnn_array.shape[0])


# In[7]:


#  extract_feat_for_video("2_Egg_Vs_95_Egg")


# In[9]:


processed = os.listdir('cnn')

for label in os.listdir('labels'):
    name = label.replace(".json", "")
    if name + ".feat" in processed:
        continue
        
    print(name)
    extract_feat_for_video(name)

