#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:33:06 2018

@author: Arpit
"""

import tensorflow as tf
import math
import operator
from utils import get_mini_batch, to_label
from graph_plot import GraphPlot
import numpy as np

input_size = 128
genre_size = 10
conv_layers = [
	{'filters':32, 'kernel_size': (8, 8), 'stride':2}
	 , {'filters':32, 'kernel_size': (8, 8), 'stride':2}
	 , {'filters':32, 'kernel_size': (4, 4), 'stride':4}
	]
fully_conn_layer = 1024

class Model():
    def __init__(self, load_model=False):
        self.graph_plot = GraphPlot("Performance", "Steps", "% Accuracy")
        self.graph_plot.addPlot(0, "training")
        self.graph_plot.addPlot(1, "validation")

        # initialising new session
        self.sess = tf.Session()

        if load_model:
            self.load_model()
        else:
            self.init_weights()
            self.init_graph()
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
#            self.saver = tf.train.Saver()

    def init_weights(self):
        print("Initializing weights.")
        self.W = []
        self.B = []
        input_channels = 1
        output_size = input_size
        
        for idx, layer in enumerate(conv_layers):
            output_channels = layer['filters']
            output_size /= layer['stride']
            if 'max_pool' in layer:
                output_size /= layer['max_pool']

            k_size = layer['kernel_size']
            
            Wi = tf.Variable(tf.truncated_normal([k_size[0], k_size[1], input_channels, output_channels], stddev=0.1))
            Bi = tf.Variable(tf.ones([output_channels])/10)
            input_channels = output_channels
            print("Weight with shape {} initialized".format(Wi.shape))
            
            self.W.append(Wi)
            self.B.append(Bi)
    
        output_pixels = int((output_size ** 2) * output_channels)
    
        Wi = tf.Variable(tf.truncated_normal([output_pixels, fully_conn_layer], stddev=0.1))
        Bi = tf.Variable(tf.ones([fully_conn_layer])/10)
        self.W.append(Wi)
        self.B.append(Bi)
    
        Wi = tf.Variable(tf.truncated_normal([fully_conn_layer, 10], stddev=0.1))
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
            
            if 'max_pool' in layer:
                k = layer['max_pool']
                Y = tf.nn.max_pool(Y, ksize=[1, k, k, 1], padding='SAME')
                
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
        print("Training model.")

        for i in range(iterations):
            batch_X, batch_Y = get_mini_batch(X_train, Y_train)
            acc, loss = self.sess.run([self.accuracy, self.cross_entropy], 
                                 feed_dict={self.X: batch_X, self.Y_: batch_Y,
                                            self.step: i, self.pkeep: 1.0})
            print("Step: {}, Accuracy: {}, Loss: {}".format(i, acc, loss))
            self.graph_plot.addData((i, acc), 0)
            
            self.sess.run(self.minimize, feed_dict={self.X: batch_X, self.Y_: batch_Y, 
                                               self.step: i, self.pkeep: 0.75})
            
            if i%10 == 0 and len(X_valid) != 0:
                acc = self.check_accuracy(X_valid, Y_valid)
                print("\nTesting data, Accuracy: {}\n".format(acc))
                self.graph_plot.addData((i, acc), 1)
                self.graph_plot.plot()
                
#            if i%100 == 0:
#                self.save_model(i)

    def check_accuracy(self, X, Y):
        preds = self.predict(X)
        accuracy = sum([1 if np.argmax(Y[i])==preds[i] else 0 for i in range(len(Y))])/len(Y)
        return accuracy
    
    def predict(self, X):
        print("Predicting.")
        
        preds = []
        for sample_images in X:
            votes = {}
            sample_images = np.array(sample_images)
            sample_preds = self.sess.run(self.Y, feed_dict={self.X: sample_images,
                                                            self.pkeep: 1.0})
            for pred in sample_preds:
                pred = np.argmax(pred)
                if pred in votes:
                    votes[pred] += 1
                else:
                    votes[pred] = 1
            
            pred = max(votes.items(), key=operator.itemgetter(1))[0]
            preds.append(pred)

        return preds

    def load_model(self):
        filename = self.get_latest_model()
        self.saver = tf.train.import_meta_graph(filename)
        self.saver.restore(self.sess, tf.train.latest_checkpoint('saved_models/'))
        graph = tf.get_default_graph()
        variables = graph.get_collection('trainable_variables')

        #loading weights
        self.W = []
        self.B = []
        i = 0        
        while i < len(variables):
            self.W.append(variables[i])
            i += 1
            self.B.append(variables[i])
            i += 1
            
        #loading tensors
        
    
    def save_model(self, i=0):
        self.saver.save(self.sess, 'saved_models/trained_model', global_step=i, max_to_keep=4)
        
    def get_latest_model(self):
        pass