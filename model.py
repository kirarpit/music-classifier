#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:33:06 2018

@author: Arpit
"""

import tensorflow as tf
import math
from utils import get_mini_batch

input_size = 128
genre_size = 10
conv_layers = [
	{'filters':32, 'kernel_size': (8, 8), 'stride':2}
	 , {'filters':32, 'kernel_size': (8, 8), 'stride':2}
	 , {'filters':32, 'kernel_size': (4, 4), 'stride':4}
	]

def init_weights():
    print("Initializing weights.")
    W = []
    B = []
    input_channels = 1
    output_size = input_size
    
    for idx, layer in enumerate(conv_layers):
        output_channels = layer['filters']
        output_size /= layer['stride']
        k_size = layer['kernel_size']
        
        Wi = tf.Variable(tf.truncated_normal([k_size[0], k_size[1], input_channels, output_channels], stddev=0.1))
        Bi = tf.Variable(tf.ones([output_channels])/10)
        input_channels = output_channels
        print("Weight with shape {} initialized".format(Wi.shape))
        
        W.append(Wi)
        B.append(Bi)

    output_pixels = int((output_size ** 2) * output_channels)

    Wi = tf.Variable(tf.truncated_normal([output_pixels, 200], stddev=0.1))
    Bi = tf.Variable(tf.ones([200])/10)
    W.append(Wi)
    B.append(Bi)

    Wi = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    Bi = tf.Variable(tf.ones([10])/10)
    W.append(Wi)
    B.append(Bi)
    
    return W, B
    
def get_next_weights(W, B, i):
    return W[i+1], B[i+1], i+1
    
def init_graph():
    X_init = tf.placeholder(tf.float32, [None, input_size, input_size, 1])
    Y_ = tf.placeholder(tf.float32, [None, genre_size])
    step = tf.placeholder(tf.int32)

    W, B = init_weights()
    
    X = X_init
    for idx, layer in enumerate(conv_layers):
        stride = layer['stride']
        Y = tf.nn.relu(tf.nn.conv2d(X, W[idx], strides=[1, stride, stride, 1], padding='SAME') + B[idx])
        print(Y.shape)
        X = Y

    Wi, Bi, idx = get_next_weights(W, B, idx)
    Y = tf.reshape(Y, shape=[-1, Wi.shape[0].value])
    Y = tf.nn.relu(tf.matmul(Y, Wi) + Bi)
    
    Wi, Bi, idx = get_next_weights(W, B, idx)
    Ylogits = tf.matmul(Y, Wi) + Bi
    Y = tf.nn.softmax(Ylogits)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) )
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
    minimize = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    return X_init, Y_, step, accuracy, cross_entropy, minimize

def train(X_train, Y_train, X_test, Y_test):
    X, Y_, step, accuracy, cross_entropy, minimize = init_graph()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(10000):
        batch_X, batch_Y = get_mini_batch(X_train, Y_train)
        acc, loss = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, step: i})
        print("Step: {}, Accuracy: {}, Loss: {}".format(i, acc, loss))
        
        sess.run(minimize, feed_dict={X: batch_X, Y_: batch_Y, step: i})
        
        if i%10 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={X: X_test, Y_: Y_test})
            print("Testing data, Accuracy: {}, Loss: {}".format(acc, loss))


