#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn import preprocessing
from sklearn.svm.classes import SVC
from matplotlib import pyplot as plt

import numpy as np
import os

from keras import models, optimizers, layers


# In[10]:


labels = os.listdir("labels")
names = []
for l in labels:
    names.append(l.replace(".json", ""))

train_names = []
test_names = ['2_Egg_Vs_95_Egg', '7_Secret_Menu_Vs_2500_Secret_Menu']
for name in names:
    if not name in test_names:
        train_names.append(name)

print("train", train_names)
print("test", test_names)


# In[11]:


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


# In[12]:


train_X, train_Y = [], []
for train_name in train_names:
    X, Y = load_data_from_video(train_name)
    train_X += X
    train_Y += Y

train_X = np.array(train_X)
train_Y = np.array(train_Y)


# In[13]:


test_X, test_Y = [], []
for test_name in test_names:
    X, Y = load_data_from_video(test_name)
    test_X += X
    test_Y += Y

test_X = np.array(test_X)
test_Y = np.array(test_Y)


# In[14]:


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


# In[15]:


# LSTM?
# neural network?

model = models.Sequential()
model.add(layers.Dense(32, input_dim=2048, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x=train_X, 
                    y=train_Y,
                    validation_data=(test_X, test_Y),
                    epochs=10, 
                    batch_size=32,
                   )


# In[16]:


# Save model
model.save('models/scene_nn.h5')


# In[17]:


def test_on_video(name, model):
    cnn_array = np.genfromtxt('cnn/' + name + '.feat', delimiter=";")
    scene_f = open('scene_labels/' + name + '.labels')
    labels = scene_f.readlines()
    scene_array = []
    for l in labels:
        scene_array.append(l.replace("\n",""))
    
    X = []
    Y = []
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


get_ipython().system('ls models')

