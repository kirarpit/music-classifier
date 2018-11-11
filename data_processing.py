#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:55:13 2018

@author: Arpit
"""
import os.path
import numpy as np
from PIL import Image
import glob

global_labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def get_training_data(files):
    """Opens the spectrogram file for given
    songs as images, converts them to np.arrays
    and one-hot encodes the label"""
    
    X = []
    Y = []
    for filename in os.listdir('images'):
        file_key = '.'.join(filename.split('.')[:2]) + '.png'
        if file_key not in files:
            continue
        
        # opens the spectrogram image for a song
        image = Image.open('images/' + filename)
        image = np.asarray(image, dtype=np.uint8)
        image = image[:, :, np.newaxis]/255
        X.append(image)
        
        # one-hot encodes the label for a song which is derived from file name
        genre = filename.split('.')[0]
        label = [0]*10
        label[global_labels.index(genre)] = 1
        label = np.array(label)
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def load_data(files):
    """One song is split into 10 splices and then for 
    each splice spectrogram is saved in a file. This
    function loads all the spectrograms for a song and
    then append the result to a list.
    This format is used to make prediction and voting
    easily.
    """
    result = []
    sample_images = []
    keys = []
    labels = []
    
    for filename in sorted(files):
        """for a song 'country.00018.5.au', 
        file_id would be 'country.00018.au' and
        file_sub_id would be '5'."""
        
        filename_split = filename.split('.')
        file_id = '.'.join(filename_split[:2]) + '.au'
        file_sub_id = filename_split[2]
        
        image = Image.open('images/' + filename)
        image = np.asarray(image, dtype=np.uint8)
        image = image[:, :, np.newaxis]/255
        sample_images.append(image)

        if file_sub_id == '9':
            result.append(sample_images)
            keys.append(file_id)
            
            label = [0]*10
            if filename_split[0] in global_labels:
                label[global_labels.index(filename_split[0])] = 1
            labels.append(label)
            
            sample_images = []
            
    return result, labels, keys

def get_validation_data(files):
    """wrapper for loading validation data
    on the function load_data"""
    new_files = []
    for file in list(files):
        for i in range(10):
            x = file.split('.')
            x[2] = str(i)
            x.append('png')
            new_files.append('.'.join(x))
    
    data, labels, _ = load_data(new_files)
    return (data, labels)

def get_testing_data():
    files = glob.glob("images/validation*.png")
    files = [file.split('/')[1] for file in files]
    data, _, keys = load_data(files)
    
    return (data, keys)

def get_image_data(data_type, files=None):
    """wrapper for loading 3 different types
    of data, i.e., training, validation and testing."""
    if data_type == "train":
        data = get_training_data(files)
    elif data_type == "valid":
        data = get_validation_data(files)
    else:
        data = get_testing_data()
    
    return data

def splice_images():
    """This function splices a spectrogram into 10
    smaller pieces. This function is not called by
    the code but was run once manually."""
    for filename in os.listdir('spectrograms'):
        img = Image.open('spectrograms/' + filename)
        filename = filename.split('.')
        filename.append('png')
        
        for i in range(10):
            filename[2] = str(i)
            starting_pixel = i * 128
            temp_image = img.crop((starting_pixel, 1, starting_pixel + 128, 129))
            temp_image.save('images/' + '.'.join(filename))