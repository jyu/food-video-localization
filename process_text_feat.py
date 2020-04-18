#!/usr/bin/env python
# coding: utf-8

# In[31]:


from pytube import YouTube
import spacy
import srt
import en_core_web_sm
import cv2
import pytesseract 
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn import preprocessing
import json
from PIL import Image
import numpy as np
import os


# In[3]:


scene_model = load_model('models/scene_nn.h5')


# In[17]:


def run_on_video(name, model):
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    video_cap = cv2.VideoCapture('data/' + name + '.mp4')
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    preds = []
    for i in range(len(cnn_array)):
        feat = cnn_array[i]
        feat = preprocessing.scale(feat)
        feat = np.array([feat])
        pred = model.predict(feat)[0]
        pred_label = "transition"
        if pred >= 0.5:
            pred_label = "scene"
        preds.append(pred_label)
    return preds, fps

# Post processing into timestamps of scenes
def get_scene_transitions(seq):
    prev_label = seq[0]   
    transitions = []
    for i in range(len(seq)):
        label = seq[i]
        
        # Transition
        if label != prev_label:
            # Check if real transition: before has to be same, after has to be same
            # 10 key frames -> 100 frames -> ~3s
            check_const = 10
            same = True
            for j in range(1,check_const):
                same = same and seq[i-1] == seq[i-j] and seq[i] == seq[i+j]

            if same:
                transitions.append((i, label))
        prev_label = label
    
    return transitions

def get_scene_timestamps(transitions, fps):
    scenes = []
    framestamps = []
    for i in range(len(transitions) - 1):
        transition = transitions[i]
        next_transition = transitions[i+1]
        if transition[1] == 'scene' and next_transition[1] == 'transition':
            scenes.append((transition[0] * 10 / fps, next_transition[0] * 10 / fps))
            framestamps.append((transition[0], next_transition[0]))
    # Get first three scenes
    scenes = scenes[:3]
    framestamps = framestamps[:3]
    return scenes, framestamps

def get_timestamps_from_video(name, model):
    preds, fps = run_on_video(name, model)
    transitions = get_scene_transitions(preds)
    timestamps, framestamps = get_scene_timestamps(transitions, fps)
    return timestamps, framestamps


# In[28]:


def get_text_from_url(url, timestamps):
    yt = YouTube(url)

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
    return timestamp_to_text


# In[33]:


def save_text_from_name(name):
    f = open('labels/' + name + '.json')
    config = json.load(f)
    f.close()
    timestamps, framestamps = get_timestamps_from_video(name, scene_model)
    timestamps_to_text = get_text_from_url(config['url'], timestamps)
    for i in timestamps_to_text:
        g = open('transcripts/' + name + '_' + str(i) + '.txt', 'w')
        g.write(timestamps_to_text[i])
        g.close()


# In[34]:


for label in os.listdir('labels'):
    name = label.replace(".json", "")
    print(name)
    save_text_from_name(name)

