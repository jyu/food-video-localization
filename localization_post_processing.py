#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pycountry
import re


# In[2]:


# US states
US_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

def is_country(s):
    if len(s) <= 3:
        return False
    try:
        countries = pycountry.countries.search_fuzzy(s)
        
        if name == countries[0].name:
            return True
        
        return False
    except:
        return False


# In[3]:


def postProcessLocation(text):
    text = text[-2:]
    text = ', '.join(text)
    
    out_t = ""
    for i in range(len(text)):
        t = text[i]
        if t.isalnum() or t in [' ', ',', '$']:
            out_t += t
    text = out_t
    text = text.replace(' I ', ' | ').replace(' 1 ', ' | ')
    return text


# In[4]:


def isValidLocation(texts):
    if len(texts) == 0:
        return False
    
    if len(texts) < 3:
        return False
    
    # Filter out completely non alphanumeric and not $ strs out
    
    # Make it alphanumeric
    loc_1 = re.sub(r'\W+', '', texts[-2])
    loc_2 = re.sub(r'\W+', '', texts[-1])
    if len(loc_1) <= 3 or len(loc_2) <= 3:
        return False
    
    has_loc = False
    for pos in texts[-1].split(' '):
        if pos in US_states:
#             print(pos, "is a US state, True")
            has_loc = True
        if is_country(pos):
#             print(pos, "is a country, True")
            has_loc = True
        
    has_price = False
    if "$" in texts[-3:][0]:
#         print("$ in ",texts[-3:][0])
        has_price = True
        
    if has_price and has_loc:
        return True
    return False


# In[5]:


def isLooseValidLocation(texts):
    if len(texts) == 0:
        return False
    
    if len(texts) < 3:
        return False
    
    # Filter out completely non alphanumeric and not $ strs out
    
    # Make it alphanumeric
    loc_1 = re.sub(r'\W+', '', texts[-2])
    loc_2 = re.sub(r'\W+', '', texts[-1])
    if len(loc_1) <= 3 or len(loc_2) <= 3:
        return False
    
    has_loc = False
    for pos in texts[-1].split(' '):
        if pos in US_states:
#             print(pos, "is a US state, True")
            has_loc = True
        if is_country(pos):
#             print(pos, "is a country, True")
            has_loc = True
        
    has_price = False
    if "$" in texts[-3:][0]:
#         print("$ in ",texts[-3:][0])
        has_price = True
        
    if has_price or has_loc:
        return True
    return False


# In[6]:


def processPredLocations(pred, save):
    print(pred)
    num_found = 0
    with open('all_pred_locations/' + pred) as json_file:
        data = json.load(json_file)

    with open('labels/' + pred) as json_file:
        labels = json.load(json_file)

    pred_framestamps = labels['pred_framestamps']
    frame_ends = []
    frame_starts = []
    scenes_to_frames = {}
    for i in range(len(pred_framestamps)):
        f = pred_framestamps[i]
        frame_ends.append(f[1] * 10)
        frame_starts.append(f[0] * 10)
        scenes_to_frames[i] = []

    scene_i = 0
    frames = sorted(map(lambda x: int(x), data.keys()))

    out = {
        'scene_i': [],
        'frames': [],
        'locations': [],
    }

    for frame in frames:
        if scene_i < len(frame_ends) and frame > frame_ends[scene_i]:
            scene_i += 1 
        scenes_to_frames[min(scene_i, len(scenes_to_frames) - 1)].append(frame)

        if scene_i > len(frame_ends):
            break

    for scene_i in scenes_to_frames:
        frames = scenes_to_frames[scene_i]
        found = False
        for isValid in [isValidLocation, isLooseValidLocation]:
            for frame in frames:
                for loc in data[str(frame)]:
                    if isValid(loc):
                        loc = postProcessLocation(loc)
                        num_found += 1
                        found = True
                        
                        print(loc, isValid)
                        loc = loc.replace(" ", "+")
                        print("https://www.google.com/maps/?q=" + loc)

                        out['locations'].append(loc)
                        out['scene_i'].append(scene_i)
                        out['frames'].append(frame)
            
                        break
                if found:
                    break
            if found:
                break
   
    print("")
    if save:
        with open('pred_locations/' + pred, 'w') as outfile:
            json.dump(out, outfile)
    return num_found


# In[8]:


# Run location post processing
predictions = os.listdir('all_pred_locations')
num_found = 0
for pred in predictions:
    num_found += processPredLocations(pred, True)
print('num found', num_found)

