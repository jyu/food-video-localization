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

import tensorflow as tf
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import gc


# In[2]:


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# In[3]:


labels = os.listdir("labels")
names = []
for l in labels:
    names.append(l.replace(".json", ""))


# In[4]:


def load_data_from_video(name):
#     print('cnn/' + name + '.feat')
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_f = open('scene_labels/' + name + '.labels')
    labels = scene_f.readlines()
    scene_array = []
    for l in labels:
        scene_array.append(l.replace("\n",""))
        
#     print("cnn array dim", cnn_array.shape, "scene array dim", len(scene_array))
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
        if scene_array[i] in ["scene", "miniscene"]:
            Y.append(1)
        else:
            Y.append(0)
    
    # Not all 0s or all 1s
    assert(sum(Y) != 0 and sum(Y) != len(Y))
    return (X, Y)


# In[5]:


video_to_data = {}
i = 0
for name in names:
    i += 1
    X, Y = load_data_from_video(name)
    video_to_data[name] = (X, Y)
    print(str(i) + '/' + str(len(names)))


# In[6]:


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

    mc = ModelCheckpoint('models/scene_nn_' + suffix + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, mc, model.count_params()


# In[10]:


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
                    epochs=30, 
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                   )
    val_max = max(history.history['val_accuracy'])
    del model
#     i = np.argmax(np.array(history.history['val_acc']))
#     train_max = history.history['accuracy'][i]
    return val_max, size


# In[11]:


def cross_validate_method(train_fn, params):
        
    # Cross validation
    k = 5
    random.shuffle(names)
    batches = len(names) // k
    # save the cross validation videos
    f = open('cross_val.txt', 'w')
    for n in names:
        f.write(str(n) + '\n')
    f.close()
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
#             X, Y = load_data_from_video(train_name)
            train_X += X
            train_Y += Y
            del X
            del Y

        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        test_X, test_Y = [], []
        for test_name in test_names:
            X, Y = video_to_data[test_name]
#             X, Y = load_data_from_video(test_name)
            test_X += X
            test_Y += Y
            del X
            del Y

        test_X = np.array(test_X)
        test_Y = np.array(test_Y)

        val_acc, size = train_fn(params, str(i), train_X, train_Y, test_X, test_Y)
        print('max val acc', val_acc)
        val_accs.append(val_acc)
        
        del train_X
        del train_Y
        del test_X
        del test_Y
        gc.collect()
    
    return np.mean(np.array(val_accs)), size


# In[ ]:

# NN
for use_batch_norm in [True, False]:
    for use_dropout in [True, False]:
        for layers in [6, 12]:
            for h in [64]:
                    
                params = {
                    'hidden_units': h,
                    'use_dropout': use_dropout,
                    'use_batch_norm': use_batch_norm,
                    'layers': layers,
                    'save': False,
                }
                f = open('out.txt', 'r')
                completed = list(f.readlines())
                if 'Params: ' + str(params) + "\n" in completed:
                  continue
                f.close()

                print('params', params)
                val_acc, size = cross_validate_method(train_nn, params)
                print(val_acc, size)
                f = open('out.txt', 'a')
                f.write('Params: ' + str(params) + '\n')
                f.write('val acc: ' + str(val_acc) + '\n')
                f.write('model size: ' + str(size) + '\n')
                f.close()

# SVM
def train_svm(params, suffix, train_X, train_Y, test_X, test_Y):
    C = params['C']
    kernel = params['kernel']
    model = SVC(gamma='scale', probability=True, C=C, kernel=kernel)
    print("Params C:", C, "kernel:", kernel)
    model.fit(train_X, train_Y)
    print("Train score", model.score(train_X, train_Y))
    test_score = model.score(test_X, test_Y)
    print("Test score", test_score)
    return test_score, None

# SVM 
Cs = [0.1, 1, 10, 100, 1000]
# kernels = ['poly', 'rbf', 'sigmoid']
kernels = ['rbf']

scores = []
for C in Cs:
    for kernel in kernels:
        params = {
            'C': C,
            'kernel': kernel,
        }
        val_acc, size = cross_validate_method(train_svm, params)
        scores.append(test_score)
