#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:43:52 2019

@author: mandeepsingh
"""

import numpy as np
import random
x = np.random.randn(10,3)
tmp = np.max(x, axis=1)
x -= tmp.reshape((x.shape[0], 1))
x = np.exp(x)
tmp = np.sum(x, axis=1)
x /= tmp.reshape((x.shape[0], 1))


np.sum(np.dot(x,x[1]))
y = np.array([[1,1,1],[2,2,3],[1,1,1]])
np.sum(np.log(np.dot(y[0],y[[1,2]].T)))

np.outer(y[0],y[1])


def sigmoid(x):
    ### YOUR CODE HERE
    s = 1/(1+np.exp(-x))

    ### END YOUR CODE

    return s


np.sum(np.log(sigmoid(-np.dot(y[0],y[[1,2]].T))))

l = np.array([1,2,4,1,2])
idx = np.where(l==1)
idx.tolist()
np.uniqu([1,2,4,1,2])

f1 = lambda x: (np.sum(x ** 2), x * 2)