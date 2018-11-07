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

class Model():
    def __init__(self):
        self.init_weights()
        self.init_graph()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def init_weights(self):
        print("Initializing weights.")
        self.W = []
        self.B = []
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
            
            self.W.append(Wi)
            self.B.append(Bi)
    
        output_pixels = int((output_size ** 2) * output_channels)
    
        Wi = tf.Variable(tf.truncated_normal([output_pixels, 200], stddev=0.1))
        Bi = tf.Variable(tf.ones([200])/10)
        self.W.append(Wi)
        self.B.append(Bi)
    
        Wi = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
        Bi = tf.Variable(tf.ones([10])/10)
        self.W.append(Wi)
        self.B.append(Bi)
        
    def get_next_weights(self, i):
        return self.W[i+1], self.B[i+1], i+1
        
    def init_graph(self):
        print("Initializing graph.")

        self.X = tf.placeholder(tf.float32, [None, input_size, input_size, 1])
        self.Y_ = tf.placeholder(tf.float32, [None, genre_size])
        self.step = tf.placeholder(tf.int32)
        self.pkeep = tf.placeholder(tf.float32)
    
        Y = self.X
        for idx, layer in enumerate(conv_layers):
            stride = layer['stride']
            Y = tf.nn.relu(tf.nn.conv2d(Y, self.W[idx], strides=[1, stride, stride, 1], padding='SAME') + self.B[idx])
            print(Y.shape)
    
        Wi, Bi, idx = self.get_next_weights(idx)
        Y = tf.reshape(Y, shape=[-1, Wi.shape[0].value])
        Y = tf.nn.relu(tf.matmul(Y, Wi) + Bi)
        Y = tf.nn.dropout(Y, self.pkeep)
        
        Wi, Bi, idx = self.get_next_weights(idx)
        Ylogits = tf.matmul(Y, Wi) + Bi
        self.Y = tf.nn.softmax(Ylogits)
    
        # cross-entropy loss function (= -sum(Y_i * log(Yi)) )
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(cross_entropy)*100
    
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        lr = 0.0001 +  tf.train.exponential_decay(0.003, self.step, 2000, 1/math.e)
        self.minimize = tf.train.AdamOptimizer(lr).minimize(self.cross_entropy)
    
    def train(self, X_train, Y_train, X_valid, Y_valid, iterations=20):
        for i in range(iterations):
            batch_X, batch_Y = get_mini_batch(X_train, Y_train)
            acc, loss = self.sess.run([self.accuracy, self.cross_entropy], 
                                 feed_dict={self.X: batch_X, self.Y_: batch_Y,
                                            self.step: i, self.pkeep: 1.0})
            print("Step: {}, Accuracy: {}, Loss: {}".format(i, acc, loss))
            
            self.sess.run(self.minimize, feed_dict={self.X: batch_X, self.Y_: batch_Y, 
                                               self.step: i, self.pkeep: 0.75})
            
            if i%10 == 0 and len(X_valid) != 0:
                acc, loss = self.sess.run([self.accuracy, self.cross_entropy], 
                                     feed_dict={self.X: X_valid, self.Y_: Y_valid,
                                                self.pkeep: 1.0})
                print("\nTesting data, Accuracy: {}, Loss: {}\n".format(acc, loss))

    def predict(self, X):
        return self.sess.run(self.Y, feed_dict={self.X: X, self.pkeep: 1.0})
