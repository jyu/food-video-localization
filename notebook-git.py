import os

# Converts jupyter notebooks into python scripts to save in git

for f in os.listdir('.'):
  if '.ipynb' == f[-6:]:
    os.system('jupyter nbconvert --to script ' + f)
