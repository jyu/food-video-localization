#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Detect if a frame is a text scene or not

from sklearn import preprocessing
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import random
import keras
import tensorflow as tf
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow.keras.backend as K

import json
import gc

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# In[7]:


labels = os.listdir("loc_frame_labels")
names = []
for l in labels:
    names.append(l.replace(".json", ""))    
print(len(names))


# In[8]:


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


# In[9]:


# For testing frame boundaries
def view_text_frames_from_video(name):
    print(name)
    f = open('loc_frame_labels/' + name + '.json')
    data = json.load(f)
    location_frames = data['label_frames']
    
    # Only trian on videos with all frames labelled
    if len(location_frames) != 3:
        return
    
    frames_to_show = []
    for f in location_frames:
#         frames_to_show.append(f - 50)
        frames_to_show.append(f)
#         frames_to_show.append(f + 50)
#         frames_to_show.append(f + 100)
        
    frame_gen = get_keyframes('data/' + name + '.mp4', 10)
    for (img, frame) in frame_gen:
        if frame in frames_to_show:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(img)
            plt.text(1500,300,"frame:\n" + str(frame))
            plt.text(1500,600,"name:\n" + str(name))


# In[10]:


def load_data_from_video(name):
    f = open('loc_frame_labels/' + name + '.json')
    data = json.load(f)
    location_frames = data['label_frames']
    
    # Only train on videos with all frames labelled
    if len(location_frames) != 3:
        return
    
    # Frames too close to the location frame so we drop them
    frame_boundaries = []
    for f in location_frames:
        frame_boundaries.append((f-50, f+100))
    
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    
    frames = 0
    X = []
    Y = []
    for i in range(cnn_array.shape[0]):
        
        keep = True
        for b in frame_boundaries:
            if frames > b[0] and frames < b[1]:
                keep = False
                
            # also downsample negative frames, get every 100th frame
            if frames % 100 != 0:
                keep = False
            
        if frames in location_frames:
#             print("using frame", frames, "location")
            X.append(cnn_array[i])
            Y.append(1)
        elif keep:
#             print("using frame", frames, "no location")
            X.append(cnn_array[i])
            Y.append(0)

        frames += 10

    # Not all 0s or all 1s
    assert(sum(Y) != 0 and sum(Y) != len(Y))
    
    return X, Y


# In[11]:


video_to_data = {}
i = 0
for name in names:
    i += 1
    res = load_data_from_video(name)
    if res == None:
        continue
    X, Y = res
    video_to_data[name] = (X, Y)
    print(str(i) + '/' + str(len(names)))


# In[12]:


k_recall = tf.keras.metrics.Recall(thresholds=0.5)
k_precision = tf.keras.metrics.Precision(thresholds=0.5)


# In[13]:


def f1_m(y_true, y_pred):
    precision = k_precision(y_true, y_pred)
    recall = k_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model(params, suffix):
    hidden_units = params['hidden_units']
    use_dropout = params['use_dropout']
    use_batch_norm = params['use_batch_norm']
    num_layers = params['layers']

    model = models.Sequential()
    model.add(Dense(hidden_units, input_dim=2048, activation='relu'))
    if use_dropout:
        model.add(Dropout(0.25))
    if use_batch_norm:
        model.add(BatchNormalization())
    
    for i in range(num_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))
        if use_dropout:
            model.add(Dropout(0.25))
        if use_batch_norm:
            model.add(BatchNormalization())
            
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    mc = ModelCheckpoint('models/text_detection_nn_' + suffix + '.h5', monitor='val_recall', mode='max', verbose=1, save_best_only=True)

    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=[
                      'accuracy',
                       f1_m,
                       k_recall, 
                       k_precision,
                    ])
    return model, mc, model.count_params()


# In[14]:


class CollectCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()

    
def train_nn(params, suffix, train_X, train_Y, test_X, test_Y):
    model, mc, size = get_model(params, suffix)
    save = params['save']
    callbacks = [CollectCallback()]
    if save:
        callbacks.append(mc)
        
    history = model.fit(x=train_X, 
                    y=train_Y,
                    validation_data=(test_X, test_Y),
                    epochs=20, 
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                   )
    val_max = max(history.history['val_recall'])
    del model
    return val_max, size


# In[17]:


def cross_validate_method(train_fn, params):
        
    # Cross validation
    k = 5
    
    f = open('cross_val.txt', 'r')
    names_f = list(f.readlines())
    names = []
    for n in names_f:
        names.append(n.replace("\n", ""))
    batches = len(names) // k
    
    print(names)
    
    val_accs = []
    for i in range(k):
        train_names = []
        test_names = []
        for j in range(i * batches, (i + 1) * batches):
            test_names.append(names[j])
        for name in names:
            if not name in test_names:
                train_names.append(name)

        train_X, train_Y = [], []
        for train_name in train_names:
            X, Y = video_to_data[train_name]
            train_X += X
            train_Y += Y
            del X
            del Y

        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        test_X, test_Y = [], []
        for test_name in test_names:
            X, Y = video_to_data[test_name]
            test_X += X
            test_Y += Y
            del X
            del Y

        test_X = np.array(test_X)
        test_Y = np.array(test_Y)

        val_acc, size = train_fn(params, str(i), train_X, train_Y, test_X, test_Y)
        print('max val_recall', val_acc)
        val_accs.append(val_acc)
        
        del train_X
        del train_Y
        del test_X
        del test_Y
        gc.collect()
    
    return np.mean(np.array(val_accs)), size


# In[ ]:


params = {
    'hidden_units': 64,
    'use_dropout': True,
    'use_batch_norm': True,
    'layers': 3,
    'save': True,
}
print('params', params)
val_acc, size = cross_validate_method(train_nn, params)
print(val_acc, size)
f = open('out.txt', 'a')
f.write('Params: ' + str(params) + '\n')
f.write('val f1: ' + str(val_acc) + '\n')
f.write('model size: ' + str(size) + '\n')


# In[30]:


# Train on all data
train_X, train_Y = [], []
for train_name in video_to_data:
    X, Y = video_to_data[train_name]
    train_X += X
    train_Y += Y
    del X
    del Y
    
train_X = np.array(train_X)
train_Y = np.array(train_Y)
model, mc, size = get_model(params, "all")
save = params['save']
callbacks = [CollectCallback()]
if save:
    callbacks.append(mc)

history = model.fit(x=train_X, 
                y=train_Y,
                epochs=40, 
                batch_size=32,
                callbacks=callbacks,
                verbose=1,
               )


# In[19]:


# Evaluate
k = 5
batches = len(names) // k
for i in range(k):
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])
    
    model = load_model('models/text_detection_nn_' + str(i) + '.h5', {'f1_m': f1_m})
    for test_name in test_names:
        print(test_name)
        f = open('pred_locations/' + test_name + '.json')
        data = json.load(f)
        location_frames = data['frames']
        print('label frames', location_frames)


        cnn_array = np.genfromtxt('cnn/' + test_name + '.feat', delimiter=";")
        found_frames = []
        res = model.predict(cnn_array)

        for i in range(res.shape[0]):
            if res[i] > 0.5:
                found_frames.append(i * 10)

        print("found frames", found_frames)

        frame_gen = get_keyframes('data/' + test_name + '.mp4', 10)
        for (img, frame) in frame_gen:
            if frame in found_frames:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.imshow(img)
                plt.text(1500,300,"frame:\n" + str(frame))

    break

        


# In[ ]:


# For bootstrapping data

def bootstrap():
    labels = os.listdir("labels")
    all_names = []
    for l in labels:
        all_names.append(l.replace(".json", ""))    

    model = load_model('models/text_detection_nn_all.h5', {'f1_m': f1_m})

    # Evaluate
    done = 0
    for test_name in all_names:
        We already have data for this
        if test_name in names:
            continue
        if test_name in name_to_location_frames:
            continue
        print(test_name)

        cnn_array = np.genfromtxt('cnn/' + test_name + '.feat', delimiter=";")
        found_frames = []
        res = model.predict(cnn_array)

        for i in range(res.shape[0]):
            if res[i] > 0.5:
                found_frames.append(i * 10)

        print("found frames", found_frames)

        frame_gen = get_keyframes('data/' + test_name + '.mp4', 10)
        for (img, frame) in frame_gen:
            if frame in found_frames:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.imshow(img)
                plt.text(1500,300,"frame:\n" + str(frame))
                plt.text(1500,600,"name:\n" + str(test_name))
        done += 1

        if done >= 10:
            break

        


# In[89]:


# Bootstrapping labels
name_to_location_frames = {
    '3_Fries_Vs_100_Fries': [1050, 7460, 13930],
    '10_Cheesesteak_Vs_120_Cheesesteak': [740, 5880, 11500],
    '29_Vs_180_Family-Style_Meats': [1290, 7590, 13590],
    '3_Sushi_Vs_250_Sushi': [1170, 3950, 6780],
    '11_Salad_Vs_95_Salad': [1650, 9440, 16670],
    '10_Sushi_&_Burger_Vs_58_Sushi_&_Burger': [2080, 8530, 15810],
    '2_Peking_Duck_Vs_340_Peking_Duck': [1350, 6560, 14110],
    '4_Burrito_Vs_32_Burrito': [970, 7510, 13780],
    '7_Cake_Vs_208_Cake_•_Japan': [1020, 5740, 10680],
    '4_Burger_Vs_777_Burger': [1500, 3860, 6400],
    '2_Pizza_Vs_2000_Pizza_•_New_York_City': [2070, 7790, 15270],
    '9_Fish_Vs_140_Fish': [1130, 8600, 15300],
    '16_Steak_Vs_150_Steak_•_Australia': [1760, 6430, 13260],
    '10_Noodles_Vs_94_Noodles': [770, 8870, 15740],
    '27_Cake_Vs_1120_Cake': [1560, 6950, 12130],
    '350_Soup_Vs_29_Soup_•_Taiwan': [1210, 9080, 17950],
    '050_Dumpling_Vs_29_Dumplings_•_Taiwan': [1310, 9060,  16680],
    '5_Fried_Chicken_Sandwich_Vs_20_Fried_Chicken_Sandwich': [960, 8970, 15970],
    '7_Double_Cheeseburger_Vs_25_Double_Cheeseburger': [1620, 10560, 16490],
    '1_Cookie_Vs_90_Cookie': [1060, 7060, 12580],
    '1_Eggs_Vs_89_Eggs_•_Japan': [1720, 8200, 19250],
    '8_Toast_Vs_20_Toast': [1060, 7320, 14650],
    '1_Sushi_Vs_133_Sushi_•_Japan': [1800, 9290, 16690],
    '13_Lasagna_Vs_60_Lasagna': [1240, 6820, 12880],
    '3_Chicken_Vs_62_Chicken_•_Taiwan': [1120, 8360, 16640],
    '11_Steak_Vs_306_Steak': [1550, 5050, 8750],
    '3_Ramen_Vs_79_Ramen_•_Japan': [1400, 7130, 12040],
    '2_Curry_Vs_75_Curry': [1980, 6650, 12590],
    '18_Wine_Vs_1000_Wine': [1150, 5960, 11530],  
    '5_Pizza_Vs_135_Pizza': [1310, 4080, 7110],
    '13_Korean_Soup_Vs_88_Korean_Soup': [1210, 8140, 15030],
    '1_Coffee_Vs_914_Coffee_•_Japan': [920, 4900, 10010],
    '17_Fried_Chicken_Vs_500_Fried_Chicken': [810, 5470, 12270],
    '15_Spaghetti_Vs_143_Spaghetti': [980, 6490, 1250],
}

# Write labels
pred_locs = os.listdir('pred_locations')
for name in all_names:
    
    frames = []
    if name in name_to_location_frames:
        frames = name_to_location_frames[name]
    
    elif name + '.json' in pred_locs:
        f = open('pred_locations/' + name + '.json')
        data = json.load(f)
        location_frames = data['frames']
        frames = location_frames
        
    if -1 in frames or len(frames) != 3:
        print("ALERT", name)
        
    # Write frames into json file
    label = {
        'label_frames': frames
    }
    with open('loc_frame_labels/' + name + '.json', 'w') as outfile:
        json.dump(label, outfile)

    

