#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:17:28 2018

@author: Arpit
"""

import scipy
import scipy.io.wavfile
import numpy as np
from dnn import get_model
from utils import get_songs, split_data, label_index, to_label, save_preds
from sklearn.preprocessing import Normalizer

input_size = 10240

def get_song_data(songs):
    """trims a song 1/3rd from both start and 
    beginning, splits the remaining part into 
    multiple smaller pieces and then finally takes
    fourier transformation of those."""
    
    songs = list(songs)
    X = list()
    Y = list()
    for song in songs:
        song = '.'.join(song.split('.')[0:2]) + '.wav'
        _, raw = scipy.io.wavfile.read('wavs/' + song)
        
        # trims 1/3rd of the start and end
        size = raw.shape[0]
        raw = raw[int(size/3):int(size*2/3)]
        size = raw.shape[0]

        # one-hot encodes the label
        label = [0]*10
        label[label_index(song.split('.')[0])] = 1
        
        cnt = int(size/input_size)
        for i in range(cnt):
            X.append(abs(scipy.fft(raw[input_size*i:input_size*(i+1)])))
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

# gets all songs' names
songs = get_songs()

#splits them into training and validation sets
training_songs, validation_songs = split_data(np.array(songs), 0.85)

# gets fourier transformed data for given songs
X, Y = get_song_data(training_songs)

#fits and transforms a normalizer
norm = Normalizer(norm='l1').fit(X)
X = norm.transform(X)

# gets fourier transformed data of validation set and transforms according to training set
X_valid, Y_valid = get_song_data(validation_songs)
X_valid = norm.transform(X_valid)

# gets convolutional network and fits the data
model = get_model(input_size)
model.fit(X, Y, n_epoch=20, validation_set=(X_valid, Y_valid), show_metric=True)

#prediction with voting is performed finally
test_songs = get_songs('test')
keys = []
preds = []
for song in test_songs:
    X, _ = get_song_data([song])
    X = norm.transform(X)
    Y_ = model.predict(X)
    label = to_label(np.argmax(np.sum(Y_, axis=0)))
    preds.append(label)
    keys.append('.'.join(song.split('.')[:2]) + '.au')

save_preds(keys, preds, 'predictions_f.csv')







