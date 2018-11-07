#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:06:48 2018

@author: Arpit
"""
import os.path
import pickle
import numpy as np
from PIL import Image

labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def splice_images():
    for filename in os.listdir('spectrograms'):
        img = Image.open('spectrograms/' + filename)
        filename = filename.split('.')
        filename.append('png')
        
        for i in range(10):
            filename[2] = str(i)
            starting_pixel = i * 128
            temp_image = img.crop((starting_pixel, 1, starting_pixel + 128, 129))
            temp_image.save('images/' + '.'.join(filename))

def get_image_data():
    data_file = 'image_data.pkl'
    if os.path.exists(data_file):
        with open(data_file, 'rb') as handle:
            print("Cache found!")
            return pickle.load(handle)

    X = []
    Y = []
    for filename in os.listdir('images'):
        image = Image.open('images/' + filename)
        image = np.asarray(image, dtype=np.uint8)
        image = image[:, :, np.newaxis]/255
        X.append(image)
        
        genre = filename.split('.')[0]
        label = [0]*10
        label[labels.index(genre)] = 1
        label = np.array(label)
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)
    
    with open(data_file, 'wb') as handle:
        pickle.dump((X, Y), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X, Y

def get_mini_batch(X, Y):
    t_rows = list(np.random.choice(X.shape[0], 100, replace=False))
    return X[t_rows], Y[t_rows]

def split_data(X, Y, percent):
    t_rows = list(np.random.choice(X.shape[0], int(X.shape[0]*percent), replace=False))
    r_rows = list(set(list(range(X.shape[0])))^set(t_rows))
    
    return X[t_rows], Y[t_rows], X[r_rows], Y[r_rows]
    