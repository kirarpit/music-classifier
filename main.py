#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:53:36 2018

@author: Arpit
"""

from utils import get_image_data, split_data
import model

X, Y = get_image_data()
X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.85)
model.train(X_train, Y_train, X_test, Y_test)

#graphs
#then see if learning rate needs to be adjusted if the testing accuracy fluctuates too much
#overfitting? dropout!
