#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:53:36 2018

@author: Arpit
"""

from utils import save_preds, get_songs, split_data, to_label
from data_processing import get_image_data
from model import Model

songs = get_songs()
songs_train, songs_valid = split_data(songs, 0.85)

X_train, Y_train = get_image_data('train', songs_train)
X_valid, Y_valid = get_image_data('valid', songs_valid)

X_test, keys = get_image_data('test')

model = Model(True)
model.train(X_train, Y_train, X_valid, Y_valid, 5000)

preds = model.predict(X_test)
preds = [to_label(pred) for pred in preds]
save_preds(keys, preds, 'predictions.csv')

#colored mel spectrograms with librosa
#pixels per second to 30 and 70

#on-server
#with elu and xavier (do on-system one)
#deeper fully connected layer
#conv variations

#on-system
#load weights - analyse voting - better voting

#batch normalization
#residual layers!