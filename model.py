#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:33:06 2018

@author: Arpit
"""

import tensorflow as tf
import math
import operator
from utils import get_mini_batch, get_latest_model
from graph_plot import GraphPlot
import numpy as np

input_size = 128
genre_size = 10
conv_layers = [
        {'filters':64, 'kernel_size':(2, 2), 'stride':1, 'max_pool':2}
         , {'filters':128, 'kernel_size':(2, 2), 'stride':1, 'max_pool':2}
         , {'filters':256, 'kernel_size':(2, 2), 'stride':1, 'max_pool':2}
         , {'filters':512, 'kernel_size':(2, 2), 'stride':1, 'max_pool':2}
        ]
fully_conn_layer = 1024
pkeep = 0.5
beta = 0.2

class Model():
    """It is a class model which contains a convolutional
    neural network and exposes two main functions, i.e.,
    'train' and 'predict'"""
    
    def __init__(self, load_model=False):
        """Initialises weights, tensor graph and saver"""
        
        # plots accuracy vs training steps
        self.graph_plot = GraphPlot("accuracy", "Steps", "% Accuracy")
        self.graph_plot.addPlot(0, "training")
        self.graph_plot.addPlot(1, "validation")

        # initialising new session
        self.sess = tf.Session()

        # loads latest model if True
        if load_model:
            success = self.load_model()
        else:
            success = False
            
        if not success:
            self.init_weights()
            self.init_graph()
            
            #must be done after initializing graph as graph might have hidden variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(max_to_keep=2)

    def init_weights(self):
        """TF variables are initialised for
        the weights of the ANN"""
        
        print("Initializing weights.")
        self.W = []
        self.B = []
        input_channels = 1
        output_size = input_size
        
        # Xavier weights initialization
        initializer = tf.contrib.layers.xavier_initializer()

        for idx, layer in enumerate(conv_layers):
            output_channels = layer['filters']
            output_size /= layer['stride']
            if 'max_pool' in layer:
                output_size /= layer['max_pool']

            k_size = layer['kernel_size']
            
            Wi = tf.Variable(initializer([k_size[0], k_size[1], input_channels, output_channels]))
            Bi = tf.Variable(initializer([output_channels])/10)
  
            input_channels = output_channels
            print("Weight with shape {} initialized".format(Wi.shape))
            
            self.W.append(Wi)
            self.B.append(Bi)
    
        output_pixels = int((output_size ** 2) * output_channels)
    
        # after conv net, weights for fully connected layers are initialised
        Wi = tf.Variable(initializer([output_pixels, fully_conn_layer]))
        Bi = tf.Variable(initializer([fully_conn_layer])/10)
        self.W.append(Wi)
        self.B.append(Bi)
    
        Wi = tf.Variable(initializer([fully_conn_layer, fully_conn_layer//2]))
        Bi = tf.Variable(initializer([fully_conn_layer//2])/10)
        self.W.append(Wi)
        self.B.append(Bi)
        
        Wi = tf.Variable(initializer([fully_conn_layer//2, 10]))
        Bi = tf.Variable(initializer([10])/10)
        self.W.append(Wi)
        self.B.append(Bi)
        
    def get_next_weights(self, i):
        return self.W[i+1], self.B[i+1], i+1
        
    def init_graph(self):
        """TF placeholers are initialised for the graph"""
        print("Initializing graph.")

        self.X = tf.placeholder(tf.float32, [None, input_size, input_size, 1], name="X")
        self.Y_ = tf.placeholder(tf.float32, [None, genre_size], name="Y_")
        self.step = tf.placeholder(tf.int32, name="step")
        
        # percentage of nodes to keep at all fully connected layers
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
    
        Y = self.X
        for idx, layer in enumerate(conv_layers):
            stride = layer['stride']
            Y = tf.nn.elu(tf.nn.conv2d(Y, self.W[idx], strides=[1, stride, stride, 1], padding='SAME') + self.B[idx])
            
            if 'max_pool' in layer:
                k = layer['max_pool']
                Y = tf.nn.max_pool(Y, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
                
            print(Y.shape)
    
        # fully connected layers with dropouts
        Wi, Bi, idx = self.get_next_weights(idx)
        Y = tf.reshape(Y, shape=[-1, Wi.shape[0].value])
        Y = tf.nn.elu(tf.matmul(Y, Wi) + Bi)
        Y = tf.nn.dropout(Y, self.pkeep)
        
        # fully connected layers with dropouts
        Wi, Bi, idx = self.get_next_weights(idx)
        Y = tf.reshape(Y, shape=[-1, Wi.shape[0].value])
        Y = tf.nn.elu(tf.matmul(Y, Wi) + Bi)
        Y = tf.nn.dropout(Y, self.pkeep)

        # softmax layer for classifying
        Wi, Bi, idx = self.get_next_weights(idx)
        Ylogits = tf.matmul(Y, Wi) + Bi
        self.Y = tf.nn.softmax(Ylogits, name="Y")
    
        # cross-entropy loss function (= -sum(Y_i * log(Yi)) )
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)
        cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy")*100
    
        #L2 regularization
        regularizer = None
        for Wx in self.W:
            if regularizer is not None:
                regularizer += tf.nn.l2_loss(Wx)
            else:
                regularizer = tf.nn.l2_loss(Wx)

        self.loss = cross_entropy + beta * regularizer

        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    
        # exponential decay in the learning rate and ADAM optimizer
        lr = 0.0001 +  tf.train.exponential_decay(0.003, self.step, 2000, 1/math.e)
        self.minimize = tf.train.AdamOptimizer(lr, name="minimize").minimize(self.loss)
    
    def train(self, X_train, Y_train, X_valid, Y_valid, iterations=20):
        """Dictionary is fed in the graph to train the model"""
        print("Training model.")

        for i in range(iterations):
            batch_X, batch_Y = get_mini_batch(X_train, Y_train)
            acc, loss = self.sess.run([self.accuracy, self.loss], 
                                 feed_dict={self.X: batch_X, self.Y_: batch_Y,
                                            self.step: i, self.pkeep: 1.0})
            print("Step: {}, Accuracy: {}, Loss: {}".format(i, acc, loss))
            self.graph_plot.addData((i, acc), 0)
            
            self.sess.run(self.minimize, feed_dict={self.X: batch_X, self.Y_: batch_Y, 
                                               self.step: i, self.pkeep: pkeep})
            
            # save model and check for accuracy on the validation set every 100 steps
            if i%100 == 0: 
                if len(X_valid) != 0:
                    acc = self.check_accuracy(X_valid, Y_valid)
                    print("\nTesting data, Accuracy: {}\n".format(acc))
                    self.graph_plot.addData((i, acc), 1)
                
                self.graph_plot.plot()
            if i%1000 == 0:
                self.save_model(i)

    def check_accuracy(self, X, Y):
        preds = self.predict(X)
        accuracy = sum([1 if np.argmax(Y[i])==preds[i] else 0 for i in range(len(Y))])/len(Y)
        return accuracy
    
    def predict(self, X):
        """A dictionary is fed to the model's graph to determine predictions"""
        preds = []
        for sample_images in X:
            votes = {}
            sample_images = np.array(sample_images)
            sample_preds = self.sess.run(self.Y, feed_dict={self.X: sample_images,
                                                            self.pkeep: 1.0})
    
            # for a song, predictions of all the 10 splices are taken and
            # then voting is performed to get the final label for this song
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
        filename = get_latest_model()
        if filename is None: return False
        
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
        self.X = graph.get_tensor_by_name("X:0")
        self.Y_ = graph.get_tensor_by_name("Y_:0")
        self.step = graph.get_tensor_by_name("step:0")
        self.pkeep = graph.get_tensor_by_name("pkeep:0")
        self.Y = graph.get_tensor_by_name("Y:0")
        self.loss = graph.get_tensor_by_name("loss:0")*100
        self.accuracy = graph.get_tensor_by_name("accuracy:0")
        self.minimize = graph.get_operation_by_name("minimize")   
        
        return True
        
    def save_model(self, i=0):
        print("Saving model with global step:{}".format(i))
        self.saver.save(self.sess, 'saved_models/trained_model', global_step=i)