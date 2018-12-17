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
import matplotlib.image as mpimg

from imgaug import augmenters as iaa


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

# Data augmentation
def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    return zoom.augment_image(image)

image = img_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(2, 1, figsize=(8, 5))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original image')
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed image')

def pan(image):
    pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y':(-0.1, 0.1)})
    return pan.augment_image(image)

image = img_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Panned Image')

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

image = img_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)
fig, axs = plt.subplots(1, 2, figsize=(8, 8))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(brightness_altered_image)
axs[1].set_title('Brightness altered image ')

def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

random_index = random.randint(0, 1000)
image = img_paths[random_index]
steering_angle = steerings[random_index]
original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
fig, axs = plt.subplots(1, 2, figsize=(7, 4))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))    

# Preprocessing the data
def img_preprocess(img):
    img = mpimg.imread(img)
    img = img[60:135, :, :]
    # COnvert to YUV ColorSpace
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

image = img_paths[100]
original_img = mpimg.imread(image)
preprocessed_img = img_preprocess(image)

fig, axis = plt.subplots(2, 1, figsize=(7, 5))
fig.tight_layout()
axis[0].imshow(original_img)
axis[0].set_title('Original image')
axis[1].imshow(preprocessed_img)
axis[1].set_title('Preprocessed image')

X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis('off')
print(X_train.shape)

def nvidia_model():
    model = Sequential()
    # Our data is already normalized, so the step is skipped
    # Subsample is stride length
    model.add(Conv2D(24, (5,5), subsample=(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5,5), subsample=(2,2), activation='elu'))
    model.add(Conv2D(48, (5,5), subsample=(2,2), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    optimizer=Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer)    
    
    return model

model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=100, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')
    
# Save the model
model.save('model.h5')

from google.colab import files
files.download('model.h5')