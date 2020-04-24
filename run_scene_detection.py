#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
from sklearn.svm.classes import SVC
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import random
import gc
import json
from tensorflow.keras.models import load_model


# In[2]:


# load cross val videos
f = open('cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()


# In[3]:


# Load models
models = []
k = 5 # K-fold cross validation
for i in range(k):
    model = load_model('models/scene_nn_' + str(i) + '.h5')
    models.append(model)


# In[4]:


def test_on_video(name, model, verbose=False):
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_f = open('scene_labels/' + name + '.labels')
    labels = scene_f.readlines()
    scene_array = []
    for l in labels:
        scene_array.append(l.replace("\n",""))
    
    feats = []
    for i in range(len(cnn_array)):
        feat = cnn_array[i]
        feat = preprocessing.scale(feat)
        feats.append(feat)
    
    feats = np.array(feats)
    preds = model.predict(feats)
    
    correct = 0
    total = 0
    pred_labels = []
    for i in range(len(scene_array)):
        pred_label = "transition"
        pred = preds[i][0]
        if pred >= 0.5:
            pred_label = "scene"
        pred_labels.append(pred_label)
        
        if verbose:
            print(i, pred_label.ljust(10), scene_array[i].ljust(10))    
        
        label = ""
        if scene_array[i] in ["scene"]:
            label = "scene"
        elif scene_array[i] in ["transition", "end"]:
            label = "transition"
        if label != "":
            total += 1
            if label == pred_label:
                correct += 1
    return correct/total, pred_labels


# In[167]:


# Post processing into timestamps of scenes
# If we see a new label thresh times (thresh frames, 10 * thresh / fps s), we treat it as a transition
def get_scene_transitions(seq, scene_thresh, transtion_tresh):
    
    mode = seq[0]   
    transitions = []
    new_mode_count = 0
    transition_start = None
    for i in range(len(seq) - 10):
        label = seq[i]
        
        # Transition
        if label != mode:
            if new_mode_count == 0:
                transition_start = i
            new_mode_count += 1
            
            change_const = transition_thresh
            if label == "scene":
                change_const = scene_thresh
            
            if new_mode_count >= change_const:
                transitions.append((transition_start, label))    
                mode = label
                new_mode_count = 0
        else:
            new_mode_count = max(0, new_mode_count - 1)
            
    
    return transitions


# In[216]:


# We want to filter out miniscenes for evaluation
# They are scenes but don't have food localization info and not focus on show
name_to_miniscene_i = {
    '050_Dumpling_Vs_29_Dumplings_•_Taiwan': 2,
    '7_Double_Cheeseburger_Vs_25_Double_Cheeseburger': 1,
    '6_Sandwich_Vs_180_Sandwich': 2,
}


# In[251]:


def get_scene_timestamps(transitions, fps, name):
    scenes = []
    framestamps = []
    for i in range(len(transitions) - 1):
        transition = transitions[i]
        next_transition = transitions[i+1]
        if transition[1] == 'scene' and next_transition[1] == 'transition':
            scenes.append((transition[0] * 10 / fps, next_transition[0] * 10 / fps))
            framestamps.append((transition[0], next_transition[0]))
    
    # Get scenes longer than 1 min
    out_scenes = []
    out_framestamps = []
    for i in range(len(scenes)):
        scene = scenes[i]
        if scene[1] - scene[0] > 60:
            out_scenes.append(scene)
            out_framestamps.append(framestamps[i])
            
    scenes = out_scenes
    framestamps = out_framestamps
    
    # Remove predicted miniscenes
    if name in name_to_miniscene_i:
        i = name_to_miniscene_i[name]
        del scenes[i]
        del framestamps[i]
    
    out_scenes = []
    out_framestamps = []
    combine = []
    
    # Combine scenes if their transition is less than 10s
    if len(scenes) > 3:
        for i in range(len(scenes) - 1):
            if scenes[i+1][0] - scenes[i][1] < 10:
                combine.append((i, i+1))
                out_scenes.append((scenes[i][0], scenes[i+1][1]))
                out_framestamps.append((framestamps[i][0], framestamps[i+1][1]))
                i += 1
            else:
                out_scenes.append(scenes[i])
                out_framestamps.append(framestamps[i])

        if len(scenes) > 3 and len(combine) > 0:
            scenes = out_scenes
            framestamps = out_framestamps
        
    # If we can't get 3 scenes, get three longest scenes
#     if len(out_scenes) < 3:
#         lengths = []
#         indices = []
#         for i in range(len(scenes)):
#             s = scenes[i]
#             lengths.append(s[1] - s[0])
#             indices.append(i)
#         top_3 = sorted(zip(lengths, indices), reverse=True)[:3]
#         top_i = []
#         for top in top_3:
#             top_i.append(top[1])
#         top_i = sorted(top_i)
#         out_scenes = []
#         out_framestamps = []
#         for i in top_i:
#             out_scenes.append(scenes[i])
#             out_framestamps.append(framestamps[i])
        
    return scenes, framestamps


# In[287]:


def check_scene_timestamps(timestamps, name):
    f = open('labels/' + name + '.json')
    data = json.load(f)
    label_timestamps = data['time_stamps']
    label_scenes = []
    # Load label scene timestamps
    for i in range(len(label_timestamps)):
        t = label_timestamps[i]
        if t[1] == "scene":
            scene_end = label_timestamps[i]
            scene_end = scene_end[0].split(":")
            end_min, end_sec = int(scene_end[0]), int(scene_end[1])
            end_time = end_min * 60 + end_sec
            
            scene_start = label_timestamps[i-1]
            scene_start = scene_start[0].split(":")
            start_min, start_sec = int(scene_start[0]), int(scene_start[1])
            start_time = start_min * 60 + start_sec
            
            label_scenes.append((start_time, end_time))
    
    # If we detected different number of scenes
    scene_diff = abs(len(label_scenes) - len(timestamps))
    
    # Difference between timestamps
    diffs = []
    for i in range(min(len(timestamps), len(label_scenes))):
        start_diff = abs(timestamps[i][0] - label_scenes[i][0])
        end_diff = abs(timestamps[i][1] - label_scenes[i][1])
        diffs.append(start_diff + end_diff)
    
    # Time TP/FP/FN
    total_fp = 0
    total_fn = 0
    total_tp = 0
    for i in range(min(len(timestamps), len(label_scenes))):
        
        timestamp = timestamps[i]
        label_scene = label_scenes[i]
#         print(timestamp, label_scene)
        fp = 0
        fn = 0
        tp_start = max(label_scene[0], timestamp[0])
        tp_end = min(label_scene[1], timestamp[1])
        tp = tp_end - tp_start
        # Predicted starts before
        if timestamp[0] < label_scene[0]:
            fp += label_scene[0] - timestamp[0]
        # Predicted starts after
        if timestamp[0] > label_scene[0]:
            fn += timestamp[0] - label_scene[0]
        # Predicted ends before
        if timestamp[1] < label_scene[1]:
            fn += label_scene[1] - timestamp[1]
        # Predicted ends after
        if timestamp[1] > label_scene[1]:
            fp += timestamp[1] - label_scene[1]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
#         print(tp, fp, fn)
    
    return scene_diff, diffs, label_scenes, total_tp, total_fp, total_fn


# In[8]:


# Cross validation
batches = len(names) // k

scene_to_preds = {}
scene_to_test_acc = {}
for i in range(k):
    train_names = []
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])

    for t in test_names:
        print(t)
        test_acc, preds = test_on_video(t, models[i], verbose=False)
        print("test acc", test_acc)
        scene_to_preds[t] = preds
        scene_to_test_acc[t] = test_acc


# In[293]:


all_diffs = []
scene_thresh = 10
transition_thresh = 10
scene_diffs = 0
total_tp = 0
total_fp = 0
total_fn = 0
for name in names:
    preds = scene_to_preds[name]

    video_cap = cv2.VideoCapture('data/' + name + '.mp4')
    fps = video_cap.get(cv2.CAP_PROP_FPS)

    transitions = get_scene_transitions(preds, scene_thresh, transition_thresh)
    timestamps, framestamps = get_scene_timestamps(transitions, fps, name)

    scene_diff, diffs, label_scenes, tp, fp, fn = check_scene_timestamps(timestamps, name)
    scene_diffs += scene_diff
    all_diffs += diffs
    total_tp += tp
    total_fp += fp
    total_fn += fn
    if scene_diff > 0:
        print(name)
        print('test acc', scene_to_test_acc[name])
        print('scene diff', scene_diff)
        print('pred scene times', timestamps)
        print('label scene times', label_scenes)
        print('time diff', np.average(np.array(diffs)))
        print('')


# print(all_diffs)
print("total average", sum(all_diffs) / len(all_diffs))
print("all scene diffs", scene_diffs)
print("")
print("Total tp", total_tp)
print("Total fp", total_fp)
print("Total fn", total_fn)
recall =  total_tp / (total_tp + total_fn)
precision = total_tp / (total_tp + total_fp)
print("Recall:", recall)
print("Precision:", total_tp / (total_tp + total_fp))
print("F1:", 2 * (recall * precision) / (precision + recall))


# In[306]:


# print(all_diffs)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(all_diffs, bins=40)
plt.title('Time difference between predicted and actual scene')
plt.xlabel('Time difference (s)')
plt.ylabel('Number of scenes')


# In[285]:


scene_thresh = 10
transition_thresh = 10
name = '10_Noodles_Vs_94_Noodles'
print(name)
preds = scene_to_preds[name]

video_cap = cv2.VideoCapture('data/' + name + '.mp4')
fps = video_cap.get(cv2.CAP_PROP_FPS)

transitions = get_scene_transitions(preds, scene_thresh, transition_thresh)
# print('test acc', scene_to_test_acc[name])
# print('transitions', transitions)
timestamps, framestamps = get_scene_timestamps(transitions, fps, name)
# print('pred scene times', timestamps)

scene_diff, time_diff, label_scenes, tp, fp, fn = check_scene_timestamps(timestamps, name)
# print('label scenes', label_scenes)
# print(scene_diff, time_diff)
print(tp, fp, fn)


# In[254]:


scene_f = open('scene_labels/' + name + '.labels')
labels = scene_f.readlines()
scene_array = []
for l in labels:
    scene_array.append(l.replace("\n",""))

        
for i in range(len(preds)):
    print(i, round(i*10/fps), preds[i].ljust(10), scene_array[i].ljust(10))

