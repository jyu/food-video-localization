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
import gc
from bert_embedding import BertEmbedding

NONE_TAG = "NONE_TAG"          # 0
LOCATION_TAG = "LOCATION_TAG"  # 1
FOOD_TAG = "FOOD_TAG"          # 2

tag_to_label = {
    NONE_TAG: 0,
    LOCATION_TAG: 1,
    FOOD_TAG: 2,
}

bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased', max_seq_length=100)


# In[2]:


# load video names
f = open('cross_val.txt', 'r')
names_f = list(f.readlines())
names = []
for n in names_f:
    names.append(n.replace("\n", ""))

f.close()


# In[3]:


def token_to_label(t):
    if not t in ["LOCATION_TAG", "FOOD_TAG"]:
        t = NONE_TAG
    return tag_to_label[t]

# Space separated list on each line
def write_labels_to_file(feat, fwrite):
    line = str(token_to_label(feat[0]))
    for m in range(1, len(feat)):
        label = token_to_label(feat[m])
        line += ' ' + str(label)
    line += "\n"
    fwrite.write(line)


# In[4]:


def label_text_from_name(name):
    print("Video name:", name, "\n")
    
    f = open('labels/' + name + '.json')
    data = json.load(f)
    
    if 'entity_list' in data:
        return
    
    all_entity_list = []
    all_entity_tag_list = []
    
    for i in range(3):
        
        f = open('transcripts/' + name + '/scene_' + str(i) + '.txt')
        lines = list(f.readlines())
        if len(lines) == 1:
            return
        f.close()
        
        # Display text to label
        for j in range(len(lines)):
            l = lines[j]
            print(i, j, l.replace("\n", ""))
        
        locations = input("Locations: ")
        foods = input("Foods: ")
    
        location_list = locations.split(', ')
        food_list = foods.split(', ')
        entity_list = []
        entity_tag_list = []
        for l in location_list:
            if l == "":
                continue
            entity_list.append(l)
            entity_tag_list.append(LOCATION_TAG)
        for f in food_list:
            if f == "":
                continue
            entity_list.append(f)
            entity_tag_list.append(FOOD_TAG)
            
        all_entity_list.append(entity_list)
        all_entity_tag_list.append(entity_tag_list)
        
    data['entity_list'] = all_entity_list
    data['entity_tag_list'] = all_entity_tag_list
    print(data)
    
    with open('labels/' + name + '.json', 'w') as outfile:
        json.dump(data, outfile)


# In[ ]:


for label in os.listdir('labels'):
    name = label.replace(".json", "")
    print(name)
    label_text_from_name(name)


# In[83]:


def label_tokens_from_name(name):
    print("Video name:", name, "\n")
    os.system('mkdir -p "token_labels/' + name + '"')

    os.system('mkdir -p "token_labels/' + name + '"')
    
    for i in range(3):
        
        f = open('transcripts/' + name + '/scene_' + str(i) + '.txt')
        lines = list(f.readlines())
        if len(lines) == 1:
            return
        f.close()
        
        # Display text to label
        for j in range(len(lines)):
            l = lines[j]
            print(i, j, l.replace("\n", ""))
        
#         location = input("Location: ")
#         foods = input("Foods: ")

        locations = "BigMista's Barbecue and Sammich Shop"
        foods = "Texas style barbecue, spare ribs"
        location_list = locations.split(', ')
        food_list = foods.split(', ')
        entity_list = []
        entity_tag_list = []
        for l in location_list:
            entity_list.append(l)
            entity_tag_list.append(LOCATION_TAG)
        for f in food_list:
            entity_list.append(f)
            entity_tag_list.append(FOOD_TAG)
        
        # Get entities so we can tokenize
        result = bert_embedding.embedding(entity_list, True)
        
        # Convert tokens back into a list, they are saved as space seperated strings
        f = open('tokens/' + name + '/scene_' + str(i) + '.txt')
        scene_tokens = list(f.readlines())
        scene_token_list = []
        
        for tk in scene_tokens:
            tk = tk.replace("\n", "")
            sent_tokens = tk.split(" ")
            scene_token_list.append(sent_tokens)
        f.close()

        for entity_i in range(len(entity_list)):
            
            entity_tokens = result[entity_i][0]
            print('entity tokens', entity_tokens, len(entity_tokens))
            
            for sent_token_i in range(len(scene_token_list)):
                
                sent_tokens = scene_token_list[sent_token_i]
                
                # Look for entity in tokens
                entity_start = -1
                for k in range(len(sent_tokens) - len(entity_tokens)):
                    same = True
                    for l in range(len(entity_tokens)):
    #                     print(k, l, entity_tokens[l], t[k+l])
                        same = same and entity_tokens[l] == sent_tokens[k + l]

                    if same:
                        print("SAME ENTITY", k)
                        entity_start = k

                # We founda location to replace
                if entity_start != -1:
                    for l in range(len(entity_tokens)):
                        sent_tokens[entity_start + l] = entity_tag_list[entity_i]
                    # Write back sent tokens
                    scene_token_list[sent_token_i] = sent_tokens
                    
        print(scene_token_list)
        
        token_label_f = open('token_labels/' + name + '/scene_' + str(i) + '.txt', 'w')
        for sent_tokens in scene_token_list:
            write_labels_to_file(sent_tokens, token_label_f)

        break

# for n in names:
n = '7_BBQ_Ribs_Vs_68_BBQ_Ribs'
label_text_from_name(n)


# In[13]:


for label in os.listdir('labels'):
    name = label.replace(".json", "")
    print(name)
    save_text_from_name(name)
    break

