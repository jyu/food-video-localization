#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pytube import YouTube
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import json


# In[108]:


url = "https://www.youtube.com/watch?v=xj-6hC5GFfI"
yt = YouTube(url)

stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

pbar = tqdm(total=stream.filesize)
def progress_fn(self, chunk, *_):
    pbar.update(len(chunk))

yt.register_on_progress_callback(progress_fn)    

fps = stream.fps
name = yt.title.replace(" ", "_")
path = stream.download(output_path="data", filename=name)
name = path.split("/")[-1].replace(".mp4", "")


# In[5]:


# Timestamps are "min:sec"
def timeStampToFrame(timestamp, fps):
    minutes, seconds = timestamp.split(":")
    total_seconds = int(seconds) + int(minutes) * 60
    frame = round(total_seconds * fps)
    return frame


# In[21]:


# Format of timestamps are end of scene

timestamps = [("0:00", "intro"), 
              ("1:01", "transition"),
              ("3:59", "scene"),
              ("6:22", "transition"),
              ("10:11", "scene"),
              ("11:14", "transition"),
              ("15:38", "scene")
             ]


# In[109]:


cap = cv2.VideoCapture('data/' + name + '.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps", fps)
# Format of timestamps are end of scene

# Format of timestamps are end of scene

timestamps = [("0:00", "intro"), 
              ("1:01", "transition"),
              ("3:59", "scene"),
              ("6:22", "transition"),
              ("10:11", "scene"),
              ("11:14", "transition"),
              ("15:38", "scene")
             ]
frame_stamps = []

# Save tuple of timestamp index, frame to save
frames_to_save = []
# Mapping from timestamp index to frames captured
frames_to_view = {}

last_end = 0

for i in range(len(timestamps)):
    t = timestamps[i]
    frame = timeStampToFrame(t[0], fps)
    
    frames_to_check = [frame - 30, frame - 20, frame - 10, frame, frame + 10, frame + 20, frame + 30]
    for check in frames_to_check:
        frames_to_save.append((i, check))
    
    frames_to_view[i] = []
    
    frame_stamps.append((last_end, frame - 30, t[1]))
    last_end = frame + 30

print("frame stamps", frame_stamps)


# In[110]:


# We want to verify that the frame boundaries we have are actual scene transitions

cap = cv2.VideoCapture('data/' + name + '.mp4')
success, img = cap.read()
frames = 0
j = 0

print("starting to process video")
while success:
    if (frames % 10) == 0:
        # Passed frame stamp transition
        if j < len(frames_to_save) and frames > frames_to_save[j][1]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_to_view[frames_to_save[j][0]].append((img, frames))
            j += 1

    success, img = cap.read()
    frames += 1
    
print("processed", frames, "frames")

for i in frames_to_view:
    views = frames_to_view[i]
    fig = plt.figure()
    for j in range(len(views)):
        (img, frames) = views[j]
        fig.add_subplot(1,len(views),j+1)

        plt.imshow(img)


# In[111]:


# Save results

label = {
    'name': name,
    'url': url,
    'video_path': 'data/' + name + '.mp4',
    'frame_stamps': frame_stamps,
    'time_stamps': timestamps,
}

with open('labels/' + name + '.json', 'w') as outfile:
    json.dump(label, outfile)


# In[112]:


from pprint import pprint

json_file = 'labels/' + name + '.json'
with open(json_file) as json_file:
    data = json.load(json_file)
    pprint(data)

