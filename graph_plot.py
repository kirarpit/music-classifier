#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:10:16 2018

@author: Arpit
"""

import numpy as np
import matplotlib.pyplot as plt

class GraphPlot:
    def __init__(self, name="graph-plot", xlabel="X-axis", ylabel="Y-axis"):
        self.name = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        self.plots = {}
        self.labels = {}
        
    def addPlot(self, plot, label=None):
        self.plots[plot] = []
        
        if label is None:
            label = len(self.plots)
            
        self.labels[plot] = label
    
    def addData(self, data, plot):
        self.plots[plot].append(data)
        
    def plot(self):
        fig = plt.figure()
        
        for key,value in self.plots.items():
            plot = np.array(value)
            X = list(plot[:, 0])
            Y = list(plot[:, 1])
            plt.plot(X, Y, label=self.labels[key])
            plt.ylabel(self.ylabel)
            plt.xlabel(self.xlabel)

        plt.legend(loc = "best")
        plt.savefig("/home/ubuntu/" + self.name + 'chart.png')
        plt.close(fig)
