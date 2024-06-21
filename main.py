import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize

# Prepare data
input_dir = ''
categories = ['']
data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatted())
        labels.append(category_idx)
        
data = np.asarray(data)
labels = np.asarray(labels)

# Train / Test Split

# Train classifier

# Test performance