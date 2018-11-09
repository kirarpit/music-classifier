#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:29:12 2018

@author: Arpit
"""

#10240, 1024, 512, 256, 10
import tflearn
from tflearn.layers.estimator import regression

def get_model(input_size):
    input_layer = tflearn.input_data(shape=[None, input_size])
    
    dense1 = tflearn.fully_connected(input_layer, 5120, activation='elu', weights_init="Xavier")
    dropout1 = tflearn.dropout(dense1, 0.65)
    
    dense2 = tflearn.fully_connected(dropout1, 1024, activation='elu', weights_init="Xavier")
    dropout2 = tflearn.dropout(dense2, 0.65)
    
    dense3 = tflearn.fully_connected(dropout2, 256, activation='elu', weights_init="Xavier")
    dropout3 = tflearn.dropout(dense2, 0.65)

    softmax = tflearn.fully_connected(dropout3, 10, activation='softmax')
    
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
#    top_k = tflearn.metrics.Top_k(3)

    net = regression(softmax, optimizer='rmsprop', loss='categorical_crossentropy')
    
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model
