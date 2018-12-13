#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:21:44 2018

@author: myidispg
"""

import numpy as np
import matplotlib.pyplot as plt
import keras

np.random.seed(0)

points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

plt.scatter(X, y)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(50, input_dim=1, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))

adam = Adam(lr=0.01)

model.compile(loss='mse', optimizer=adam)
model.fit(X, y, epochs=50)
predictions = model.predict(X)
plt.scatter(X, y)
plt.plot(X, predictions, 'ro')
plt.show()