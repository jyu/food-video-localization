#!/usr/bin/env python
# coding: utf-8

# In[148]:


import os
import random
import numpy as np
import gc
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# In[75]:


# Lines of transcript to look at
NUM_LINES = 5
BERT_DIM = 1024


# In[76]:


# load video names
names = []
for n in os.listdir('token_labels'):
    n = n.replace(".json", "")
    if len(os.listdir('token_labels/' + n)) == 3:
        names.append(n)
        
print("total names", len(names))

k = 5
random.shuffle(names)
batches = len(names) // k
# save the cross validation videos
# f = open('text_cross_val.txt', 'w')
# for n in names:
#     f.write(str(n) + '\n')
# f.close()


# In[119]:


# load video names we stored already
f = open('text_cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()
print(names)


# In[78]:


# Find the longest sequence
MAX_LEN = 0
for name in names:
    for i in range(3):
        f = open('token_labels/' + name + '/scene_' + str(i) + '.txt')
        lines = f.readlines()
        f.close()
        
        # Use first 20 lines
        lines = lines[:20]
        for line in lines:
            line = line.replace("\n", "")
            MAX_LEN = max(MAX_LEN, len(line.split(" ")))


print(MAX_LEN)
       


# In[85]:


MAX_LEN = 40

# TO REDUCE MAX LEN FOR LESS PARAMS
# - REMOVE STOP WORDS
# - REMOVE (PERSON NAME) FROM TRANSCRIPTS


# In[120]:


def loadDataForVideo(name):
    total_loc_count = 0
    total_food_count = 0
    X = []
    Y = []
    for i in range(3):
#         total_lines = len(os.listdir('embeddings/' + name + '/scene_' + str(i)))
#         # Load BERT embeddings
#         for sent_i in range(min(NUM_LINES, total_lines)):
#             feat = np.genfromtxt('embeddings/' + name + '/scene_' + str(i) + '/sentence_' + str(sent_i) + '.feat', delimiter=";")
#             X.append(feat)
            
        # Load labels
        f = open('token_labels/' + name + '/scene_' + str(i) + '.txt')
        lines = list(f.readlines())
        f.close()
        
        # Use first 20 lines
        lines = lines[:NUM_LINES]
        for line_i in range(len(lines)):
            
            line = lines[line_i]
            line = line.replace("\n", "")
            labels = line.split(" ")
            labels = list(map(lambda x: int(x), labels))
            
            loc_count = 0
            food_count = 0

            for l in labels:
                if l == 1:
                    loc_count += 1
                    total_loc_count += 1
                elif l == 2:
                    food_count += 1
                    total_food_count += 1
            
            # Want to downsample sentences that have no labels
#             if loc_count == 0 and food_count == 0 and line_i % 5 != 0:
#                 continue
            
            # Padding
            labels = labels[:MAX_LEN]
            while len(labels) < MAX_LEN:
                labels.append(0)
            
            Y.append(labels)
            
            # Load BERT embeddings
            feat = np.genfromtxt('embeddings/' + name + '/scene_' + str(i) + '/sentence_' + str(line_i) + '.feat', delimiter=";")
            X.append(feat)

        
    print("For video", name, "loc count", total_loc_count, "food count", total_food_count, 'total size', len(X))
    return X, Y


# In[81]:


video_to_data = {}
i = 0
for name in names:
    i += 1
    X, Y = loadDataForVideo(name)
    video_to_data[name] = (X, Y)
    if i % 5 == 0:
        print(str(i) + '/' + str(len(names)))


# In[100]:


def get_bidirectional_lstm(params, suffix):
    hidden_units = params['hidden_units']
    inp = Input(shape=(MAX_LEN,BERT_DIM))
    model = Bidirectional(LSTM(units=hidden_units, return_sequences=True, recurrent_dropout=0.1))(inp)
    for i in range(params['layers'] - 1):
        model = Bidirectional(LSTM(units=hidden_units, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(3, activation="softmax"))(model)  # softmax output layer
    model = Model(inp, out)
    print(model.summary())
    suffix = 'h' + str(hidden_units) + '_l_' + str(layers) + '_' + suffix
    mc_l = ModelCheckpoint('models/location_text_blstm_' + suffix + '.h5', monitor='val_location_recall', mode='max', verbose=1, save_best_only=True)
    mc_f = ModelCheckpoint('models/food_text_blstm_' + suffix + '.h5', monitor='val_food_recall', mode='max', verbose=1, save_best_only=True)

    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=[
            tf.keras.metrics.Recall(class_id=1, name="location_recall"),
            tf.keras.metrics.Precision(class_id=1, name="location_precision"),
            tf.keras.metrics.Recall(class_id=2, name="food_recall"),
            tf.keras.metrics.Precision(class_id=2, name="food_precision")
        ]
    )
    return model, mc_l, mc_f, model.count_params()


# In[107]:


class CollectCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()

    
def train_lstm(params, suffix, train_X, train_Y, test_X, test_Y):
    model, mc_l, mc_f, size = get_bidirectional_lstm(params, suffix)
    callbacks = [CollectCallback(), mc_l, mc_f]
        
    history = model.fit(x=train_X, 
                    y=train_Y,
                    validation_data=(test_X, test_Y),
                    epochs=100, 
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                   )
    val_max_loc = max(history.history['val_location_recall'])
    val_max_food = max(history.history['val_food_recall'])

    return val_max_loc, val_max_food, size, model


# In[110]:


def cross_validate_lstm(params):
    batches = len(names) // k
    val_locs = []
    val_foods = []
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

        train_X = pad_sequences(train_X, maxlen=MAX_LEN)
        train_Y = np.array([to_categorical(i, num_classes=3) for i in train_Y])

        test_X, test_Y = [], []
        loc_count = 0
        food_count = 0

        for test_name in test_names:
            X, Y = video_to_data[test_name]
            test_X += X
            test_Y += Y
            for label in Y:
                 for l in label:
                    if l == 1:
                        loc_count += 1
                    elif l == 2:
                        food_count += 1
            del X
            del Y

        test_X = pad_sequences(test_X, maxlen=MAX_LEN)
    #     print(test_Y[0])
    #     print('loc', loc_count, 'food', food_count)

        test_Y = np.array([to_categorical(i, num_classes=3) for i in test_Y])

        loc_count = 0
        food_count = 0

        for j in range(test_Y.shape[0]):
            for s in range(MAX_LEN):
                pred = np.argmax(test_Y[j][s])
                if pred == 1:
                    loc_count += 1
                if pred == 2:
                    food_count += 1
    #     print('loc', loc_count, 'food', food_count)


        print(train_X.shape, train_Y.shape)
        print(test_X.shape, test_Y.shape)
        val_loc, val_food, size, model = train_lstm(params, str(i), train_X, train_Y, test_X, test_Y)
        print('max loc recall', val_loc, 'max food recall', val_food)
        val_locs.append(val_loc)
        val_foods.append(val_food)

        del train_X
        del train_Y
        del test_X
        del test_Y
        gc.collect()

    val_loc = sum(val_locs) / len(val_locs)
    val_food = sum(val_foods) / len(val_foods)
    return val_loc, val_food


# In[115]:


for layers in [3]:
    for h in [64]:
        if h == 32 and layers == 3:
            continue
        params = {
            'hidden_units': h,
            'layers': layers,
        }
        print('params', params)
        val_loc, val_food = cross_validate_lstm(params)
        print(val_acc, size)
        f = open('lstm_out.txt', 'a')
        f.write('Params: ' + str(params) + '\n')
        f.write('val loc: ' + str(val_loc) + '\n')
        f.write('val food: ' + str(val_food) + '\n')
        f.close()


# In[112]:


print(val_loc, val_food)


# In[ ]:


def load_eval_data(name):
    total_loc_count = 0
    total_food_count = 0
    X = []
    Y = []
    for i in range(3):
#         total_lines = len(os.listdir('embeddings/' + name + '/scene_' + str(i)))
#         # Load BERT embeddings
#         for sent_i in range(min(NUM_LINES, total_lines)):
#             feat = np.genfromtxt('embeddings/' + name + '/scene_' + str(i) + '/sentence_' + str(sent_i) + '.feat', delimiter=";")
#             X.append(feat)
            
        # Load labels
        f = open('token_labels/' + name + '/scene_' + str(i) + '.txt')
        lines = list(f.readlines())
        f.close()
        
        # Use first 20 lines
        lines = lines[:NUM_LINES]
        for line_i in range(len(lines)):
            
            line = lines[line_i]
            line = line.replace("\n", "")
            labels = line.split(" ")
            labels = list(map(lambda x: int(x), labels))
            
            loc_count = 0
            food_count = 0

            for l in labels:
                if l == 1:
                    loc_count += 1
                    total_loc_count += 1
                elif l == 2:
                    food_count += 1
                    total_food_count += 1
            
            # Want to downsample sentences that have no labels
#             if loc_count == 0 and food_count == 0 and line_i % 5 != 0:
#                 continue
            
            # Padding
            labels = labels[:MAX_LEN]
            while len(labels) < MAX_LEN:
                labels.append(0)
            
            Y.append(labels)
            
            # Load BERT embeddings
            feat = np.genfromtxt('embeddings/' + name + '/scene_' + str(i) + '/sentence_' + str(line_i) + '.feat', delimiter=";")
            X.append(feat)

        
    print("For video", name, "loc count", total_loc_count, "food count", total_food_count, 'total size', len(X))
    return X, Y


# In[117]:


model = load_model('models/location_text_blstm_h64_l_3_0.h5')


# In[146]:


def evaluate_model(model, name):
    res = {}
    for scene_i in range(3):
        X = []
        Y = []
        
         # Load labels
        f = open('token_labels/' + name + '/scene_' + str(scene_i) + '.txt')
        lines = list(f.readlines())
        f.close()
        
        # Use first 20 lines
        lines = lines[:NUM_LINES]
        
        lines = lines[:NUM_LINES]
        for line_i in range(len(lines)):
            
            line = lines[line_i]
            line = line.replace("\n", "")
            labels = line.split(" ")
            labels = list(map(lambda x: int(x), labels))
            
            # Padding
            labels = labels[:MAX_LEN]
            while len(labels) < MAX_LEN:
                labels.append(0)
            
            Y.append(labels)
            
            # Load BERT embeddings
            feat = np.genfromtxt('embeddings/' + name + '/scene_' + str(scene_i) + '/sentence_' + str(line_i) + '.feat', delimiter=";")
            X.append(feat)
        
        X = pad_sequences(X, maxlen=MAX_LEN)
        Y = np.array([to_categorical(i, num_classes=3) for i in Y])
        print('X shape:', X.shape, 'Y shape:', Y.shape)
        out = model.predict(X)
        print('Out shape:', out.shape)

    
    
        f = open("tokens/" + name + '/scene_' + str(scene_i) + '.txt')
        tokens = []
        for toks in f.readlines():
            tokens.append(toks.split(' '))
        locs, foods = [], []
        for i in range(X.shape[0]):
            for s in range(X.shape[1]):
                pred = np.argmax(out[i][s])
                if s < len(tokens[i]) - 1:
                    if pred == 1:
                        locs.append(tokens[i][s])
                    if pred == 2:
                        foods.append(tokens[i][s])

        print("Scene i", scene_i)
        print('Locations:', locs)
        print('Foods:', foods)
        res[scene_i] = {
            'foods': foods,
            'locations': locs
        }
        
    with open('text_preds/' + name + '.json', 'w') as outfile:
        json.dump(res, outfile)
    
# evaluate_model(model, '5_Pie_Vs_250_Pie')


# In[140]:


print(names)
print(len(names))


# In[149]:


# Save results from NER
k = 5
batches = len(names) // k
for i in range(k):
    test_names = []
    for j in range(i * batches, (i + 1) * batches):
        test_names.append(names[j])
        
    model = load_model('models/location_text_blstm_h64_l_3_' + str(i) + '.h5')
    for name in test_names:
        evaluate_model(model, name)
    

