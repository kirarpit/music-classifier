#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:06:48 2018

@author: Arpit
"""
import numpy as np
import glob

labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def get_mini_batch(X, Y):
    t_rows = list(np.random.choice(X.shape[0], 100, replace=False))
    return X[t_rows], Y[t_rows]

def split_data(X, percent):
    t_rows = list(np.random.choice(X.shape[0], int(X.shape[0]*percent), replace=False))
    r_rows = list(set(list(range(X.shape[0])))^set(t_rows))
    
    return X[t_rows], X[r_rows]
   
def save_preds(keys, preds, filename):
    f = open(filename,'w')
    f.write("id,class\n")
    
    for i in range(len(keys)):
        f.write(keys[i] + "," + preds[i] + "\n")
    f.close()
    
def to_label(idx):
    return labels[idx]

def get_songs():
    s = set()
    files = glob.glob("images/*.png")
    files = [file.split('/')[1] for file in files]
    
    for file in files:
        if file.split('.')[0] == 'validation': continue
        s.add('.'.join(file.split('.')[:2]) + '.png')
        
    return np.array(list(s))

def get_latest_model():
    files = glob.glob("saved_models/*meta")
    files.sort()
    
    return files[-1]
