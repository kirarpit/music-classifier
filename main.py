#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:53:36 2018

@author: Arpit
"""

from utils import save_preds, get_songs, split_data, to_label
from data_processing import get_image_data
from model import Model

# get names of all the songs
songs = get_songs()

# split them in training and valid sets according to given percentage
songs_train, songs_valid = split_data(songs, 0.85)

# get actual spectrogram(2d np.array) data for the songs
X_train, Y_train = get_image_data('train', songs_train)
X_valid, Y_valid = get_image_data('valid', songs_valid)

# get names and spectrogram data for the final testing set(which is to be uploaded)
X_test, keys = get_image_data('test')

model = Model(False)
model.train(X_train, Y_train, X_valid, Y_valid, 5000)

preds = model.predict(X_test)
preds = [to_label(pred) for pred in preds]
save_preds(keys, preds, 'predictions.csv')