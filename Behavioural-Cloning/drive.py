import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import cv2
import os
import random
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

datadir = 'Data/'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)
data.head()

# function to get only the image name from the complete image path
def path_leaf(path):
    # This splits the string by the final slash.
    head, tail = ntpath.split(path)
    return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

# Histogram to analyze steering angles. Helps to find the dominant steering angles.
num_bins = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5 
# ^^ Center the bins because the obtained bins are not centered at 0.
print(center)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

# Balancing the extra data
print('total data length- ', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin: ]
    remove_list.extend(list_)

print('len of data to be removed- ', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining data length- ', len(data))

hist, _ = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.05) # here centered bins are already centered bins from the initial data
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

# Training data loading
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_path = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_path, steering

img_paths, steerings = load_img_steering(datadir +'/IMG', data)

# Split into train test sets.
X_train, X_valid, y_train, y_valid = train_test_split(img_paths, steerings, test_size=0.2, random_state=6)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title("Training Set")
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title("Validation Set")

# Preprocessing the data
def img_preprocess(img):
    
