#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytube import YouTube
import srt
import cv2
from matplotlib import pyplot as plt
import json
from PIL import Image
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from bert_embedding import BertEmbedding
import gc

bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased', max_seq_length=100)


# In[2]:


# load video names
f = open('cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()


# In[10]:


def write_list_to_file(feat, fwrite):
    line = str(feat[0])
    for m in range(1, len(feat)):
      line += ' ' + str(feat[m])
    line += "\n"
    fwrite.write(line)
    
def write_features_to_file(features, fwrite):
    for i in range(features.shape[0]):
        feat = features[i]
        line = str(feat[0])
        for m in range(1, feat.shape[0]):
          line += ';' + str(feat[m])
        line += "\n"
        fwrite.write(line)


# In[4]:


def get_label_timestamps(name):
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
    return label_scenes


# In[5]:


def get_text_from_url(url, timestamps):
    yt = YouTube(url)
    caption = yt.caption_tracks[0]
    print(caption.name) 
#     print(yt.captions)
#     print(yt.captions.all())
#     caption = yt.captions.get_by_language_code('en')
    
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


# In[15]:


def save_text_from_name(name):
    f = open('labels/' + name + '.json')
    print("Video name:", name, "\n")
    config = json.load(f)
    f.close()
    timestamps = get_label_timestamps(name)
    timestamps_to_text = get_text_from_url(config['url'], timestamps)
    
    # Init directories
    os.system('mkdir -p transcripts/' + name)
    os.system('mkdir -p tokens/' + name)
    os.system('mkdir -p embeddings/' + name)

    
    transcripts = []
    for i in timestamps_to_text:
        text = timestamps_to_text[i]
        sent_text = nltk.sent_tokenize(text)
        # Test
        sent_text = [sent_text[0]]
        print("For scene", i, "found ", len(sent_text), "sentences")
        result = bert_embedding.embedding(sent_text, True)
        print("Got embedding")
        
        os.system('mkdir -p embeddings/' + name + '/scene_' + str(i))
        transcript_f = open('transcripts/' + name + '/scene_' + str(i) + '.txt', 'w')
        token_f = open('tokens/' + name + '/scene_' + str(i) + '.txt', 'w')
        
        # Save per transcript, tokens, and embedding per sentence
        for j in range(len(result)):
            transcript = sent_text[j]
            tokens = result[j][0]
            embedding = result[j][1]
            embedding_f = open('embeddings/' + name + '/scene_' + str(i) + '/sentence_' + str(j) + '.feat', 'w')

            print('embedding shape', np.array(embedding).shape)
            
            transcript_f.write(transcript + '\n')
            write_list_to_file(tokens, token_f)
            write_features_to_file(np.array(embedding), embedding_f)
            embedding_f.close()

        
        transcript_f.close()
        token_f.close()
        
        gc.collect()
        del text, sent_text, result
        
# name = '3_Seafood_Vs_213_Seafood_•_Australia'
name = '7_BBQ_Ribs_Vs_68_BBQ_Ribs'
save_text_from_name(name)


# In[ ]:


def label_text_from_name(name):
    f = open('labels/' + name + '.json')
    print("Video name:", name, "\n")
    config = json.load(f)
    f.close()
    timestamps = get_label_timestamps(name)
    timestamps_to_text = get_text_from_url(config['url'], timestamps)
    
    location = "BigMista's Barbecue and Sammich Shop"
    s = "(jazz music) - [Neil] Hey guys, my name is Neil Strawder, also known as BigMista, of BigMista's Barbecue and Sammich Shop."
    # location embedding
    result = bert_embedding.embedding([location], True)
    tokens = result[0][0]
    print('location tokens', tokens, len(tokens))

    print(s,)
    result = bert_embedding.embedding([s], True)
    tokens = result[0][0]
    print('tokens', tokens, len(tokens))
    print('embedding shape', np.array(result[0][1]).shape)
    
#     transcripts = []
#     for i in timestamps_to_text:
#         text = timestamps_to_text[i]
        

#         sent_text = nltk.sent_tokenize(text)
#         for s in sent_text:
#             print(s,)
#             result = bert_embedding.embedding([s], True)
#             tokens = result[0][0]
#             print('tokens', tokens, len(tokens))
#             print('embedding shape', np.array(result[0][1]).shape)

#         print("")
#         foods = input("Scene foods: ")
#         for food in foods.split(", "):
#             f_len = len(food.split(" "))
#             replace_str = "FOOD_TAG " * f_len
#             replace_str = replace_str[:-1]
#             text = text.replace(food, replace_str)
#         location = input("Location: ")
#         if location != "":
#             l_len = len(location.split(" "))
#             replace_str = "LOCATION_TAG " * l_len
#             replace_str = replace_str[:-1]
#             text = text.replace(location, replace_str)
#         print(text, "\n")
#         print(word_tokenize(text))

#         print("")
#         break
        
        
#         g = open('transcripts/' + name + '_' + str(i) + '.txt', 'w')
#         g.write(timestamps_to_text[i])
#         g.close()

# name = '3_Seafood_Vs_213_Seafood_•_Australia'
name = '7_BBQ_Ribs_Vs_68_BBQ_Ribs'
save_text_from_name(name)


# In[13]:


for label in os.listdir('labels'):
    name = label.replace(".json", "")
    print(name)
    save_text_from_name(name)
    break

