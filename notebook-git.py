import os

# Converts jupyter notebooks into python scripts to save in git
ignore_list = ['scene_detection.ipynb']
for f in os.listdir('.'):
  if f in ignore_list:
    continue

  if '.ipynb' == f[-6:]:
    os.system('jupyter nbconvert --to script ' + f)
