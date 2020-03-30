#!/usr/bin/env python
# coding: utf-8

# In[77]:


from pytube import YouTube
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import scenedetect
import pytesseract 
import numpy as np
from PIL import Image
import srt

import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions


# In[17]:


yt = YouTube("https://www.youtube.com/watch?v=DWq_sSSWWSI")

stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

pbar = tqdm(total=stream.filesize)
def progress_fn(self, chunk, *_):
    pbar.update(len(chunk))

print(name)
yt.register_on_progress_callback(progress_fn)    

fps = stream.fps
name = yt.title.replace(" ", "_")
stream.download(output_path="./data", filename=name+" 30")


# In[44]:


scene_list = []  
path = "data/47_Taco_Vs_1_Taco 30.mp4"

video_mgr = scenedetect.VideoManager([path])    #put your filename in this line
stats_mgr = scenedetect.stats_manager.StatsManager()
scene_mgr = scenedetect.SceneManager(stats_mgr)

# Now add the content detector
scene_mgr.add_detector(scenedetect.ContentDetector(threshold=60, min_scene_len=5 * 30))

# Start the video manager
downscale_factor=1
video_mgr.set_downscale_factor(downscale_factor)
video_mgr.start()

# Detect the scenes
scene_mgr.detect_scenes(frame_source=video_mgr, show_progress=True)


# In[46]:


# Retrieve scene list
base_timecode = 30
scene_mgr_list = scene_mgr.get_scene_list(base_timecode)
# print(scene_mgr_list)

# Initialize scene list for analysis
scene_list = []

# Build our list from the frame_timecode objects
for scene in scene_mgr_list:
    start_frame, end_frame = scene
    try:
        start_frame = start_frame.frame_num
    except:
        start_frame = 0
    scene_list.append(start_frame/30)

last_frame, end = scene_mgr_list[-1]
print(scene_list)
# Extract some info
video_fps = last_frame.framerate
frames_read = end_frame

# Release the video manager
video_mgr.releaseb()


# In[93]:


model = ResNet50(weights='imagenet')

# Takes in OpenCV image
def getResNet(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im_resized = im_pil.resize((224, 224))
    x = image.img_to_array(im_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    res = decode_predictions(preds, top=5)
    res_filtered = []
    for r in res[0]:
        res_filtered.append((r[1], r[2]))
    return res_filtered


# In[68]:


# Use YT SRT for captioning, otherwise use gcloud speech to text API
yt = YouTube("https://www.youtube.com/watch?v=DWq_sSSWWSI")

caption = yt.captions.get_by_language_code('en')
caption_srt = caption.generate_srt_captions()
subtitle_generator = srt.parse(caption_srt)

for sub in subtitle_generator:
    print(sub)
    print(sub.start.total_seconds())
    print(sub.end.total_seconds())
    print(sub.content)
    break


# In[96]:


name = "47_Taco_Vs_1_Taco 30.mp4"

fps = int(name.replace(".mp4", "").split(" ")[1])
print("fps", fps)

vidcap = cv2.VideoCapture('data/' + name)
count = 0
success = True

while success:
    success,img = vidcap.read()
    if count % fps == 0:
        resnet_res = getResNet(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        pil_img = Image.fromarray(img)
        text = pytesseract.image_to_string(pil_img)
        plt.text(1500,150,"text:\n" + str(text))
        plt.text(1500,300,"classes:\n" + str(resnet_res))


    count += 1

