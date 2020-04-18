#!/usr/bin/env python
# coding: utf-8

# In[26]:


from sklearn import preprocessing
from sklearn.svm.classes import SVC
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import random
from keras import models, optimizers, layers
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[5]:


labels = os.listdir("labels")
names = []
for l in labels:
    names.append(l.replace(".json", ""))

test_names = random.choices(names, k=4)
train_names = []
for name in names:
    if not name in test_names:
        train_names.append(name)

print("train", train_names)
print("test", test_names)


# In[6]:


def load_data_from_video(name):
    print('cnn/' + name + '.feat')
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_f = open('scene_labels/' + name + '.labels')
    labels = scene_f.readlines()
    scene_array = []
    for l in labels:
        scene_array.append(l.replace("\n",""))
        
    print("cnn array dim", cnn_array.shape, "scene array dim", len(scene_array))
    # Make sure data is same shape as label
    assert(cnn_array.shape[0] == len(scene_array))
    
    X = []
    Y = []
    for i in range(len(scene_array)):
        if not scene_array[i] in ["scene", "transition", "end"]:
            continue
        feat = cnn_array[i]
        feat = preprocessing.scale(feat)
        X.append(feat)
        if scene_array[i] == "scene":
            Y.append(1)
        else:
            Y.append(0)
    
    # Not all 0s or all 1s
    assert(sum(Y) != 0 and sum(Y) != len(Y))
    return (X, Y)


# In[7]:


train_X, train_Y = [], []
for train_name in train_names:
    X, Y = load_data_from_video(train_name)
    train_X += X
    train_Y += Y

train_X = np.array(train_X)
train_Y = np.array(train_Y)


# In[8]:


test_X, test_Y = [], []
for test_name in test_names:
    X, Y = load_data_from_video(test_name)
    test_X += X
    test_Y += Y

test_X = np.array(test_X)
test_Y = np.array(test_Y)


# In[9]:


print('train X shape', train_X.shape)
print('train Y shape', train_Y.shape)

print('test X shape', test_X.shape)
print('test Y shape', test_Y.shape)


# In[ ]:


Cs = [0.1, 1, 10, 100, 1000]
# kernels = ['poly', 'rbf', 'sigmoid']
kernels = ['rbf']

params = []
scores = []
models = []
for C in Cs:
    for kernel in kernels:
        gamma = 'scale'
        model = SVC(gamma=gamma, probability=True, C=C, kernel=kernel)
        print("Params C:", C, "kernel:", kernel)
        model.fit(train_X, train_Y)
        print("Train score", model.score(train_X, train_Y))
        test_score = model.score(test_X, test_Y)
        print("Test score", test_score)
        params.append({'C': C, 'kernel': kernel})
        scores.append(test_score)
        models.append(model)


# In[31]:


# LSTM?
# neural network?

model = models.Sequential()
model.add(layers.Dense(16, input_dim=2048, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

mc = ModelCheckpoint('models/best_scene_nn.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x=train_X, 
                    y=train_Y,
                    validation_data=(test_X, test_Y),
                    epochs=20, 
                    batch_size=32,
                    callbacks=[mc]
                   )


# In[32]:


# Save model
model.save('models/scene_nn.h5')


# In[2]:


model = load_model('models/scene_nn.h5')


# In[33]:


def test_on_video(name, model):
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_f = open('scene_labels/' + name + '.labels')
    labels = scene_f.readlines()
    scene_array = []
    for l in labels:
        scene_array.append(l.replace("\n",""))
    
    for i in range(len(scene_array)):
        feat = cnn_array[i]
        feat = preprocessing.scale(feat)
        feat = np.array([feat])
        pred = model.predict(feat)[0]
        pred_label = "transition"
        if pred >= 0.5:
            pred_label = "scene"
        print(pred_label.ljust(10), scene_array[i].ljust(10))

test_on_video('2_Egg_Vs_95_Egg', model)
    


# In[ ]:


3_Seafood_Vs_213_Seafood_•_Australia
test_on_video('2_Egg_Vs_95_Egg', model)


# In[8]:


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


# In[5]:


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


# In[10]:


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


# In[11]:


def get_timestamps_from_video(name, model):
    preds, fps = run_on_video(name, model)
    transitions = get_scene_transitions(preds)
    print(transitions)
    timestamps, framestamps = get_scene_timestamps(transitions, fps)
    print(timestamps)
    print(framestamps)
    return timestamps

get_timestamps_from_video('2_Egg_Vs_95_Egg', model)

