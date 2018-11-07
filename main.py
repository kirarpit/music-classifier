#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:53:36 2018

@author: Arpit
"""

from utils import get_image_data, split_data, load_test_data, to_label, save_preds
from model import Model
import operator
import numpy as np

X, Y = get_image_data()
X_test = load_test_data()
X_train, Y_train, X_valid, Y_valid = split_data(X, Y, 0.85)

model = Model()
model.train(X_train, Y_train, X_valid, Y_valid)

preds = []
for filename, sample_images in X_test:
    votes = {}
    sample_images = np.array(sample_images)
    sample_preds = model.predict(sample_images)

    for pred in sample_preds:
        pred = np.argmax(pred)
        if pred in votes:
            votes[pred] += 1
        else:
            votes[pred] = 1
    
    pred = max(votes.items(), key=operator.itemgetter(1))[0]
    pred_label = to_label(pred)
    print(votes, pred, pred_label)

    preds.append((filename, pred_label))

print(preds)
save_preds(preds, 'predictions.csv')

#graphs
#then see if learning rate needs to be adjusted if the testing accuracy fluctuates too much
#batch normalization
#checkpoints!
#different archi. (more like what medium guy did)
#residual layers!