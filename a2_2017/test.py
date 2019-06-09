#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:50:06 2019

@author: mandeepsingh
"""
import numpy as np

#y = np.array([[0, 1], [1, 0], [1, 0]])
#yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])


#loss = -np.sum(np.log(yhat)*y)
#result = -3 * np.log(.5)
#assert np.amax(np.fabs(loss - result)) <= 1e-6

x = np.array([[1001,1002],[3,4]])
tmp = np.max(x, axis=1)
x -= tmp.reshape((x.shape[0], 1))
x = np.exp(x)
tmp = np.sum(x, axis=1)
x /= tmp.reshape((x.shape[0], 1))
print(np.amax(np.fabs(x - np.array([0.26894142,  0.73105858]))))
assert np.amax(np.fabs(x - np.array(
      [0.26894142,  0.73105858]))) <= 1e-6
        
vocab = ['the','like','between','did','just','national','day','country','under','such','second']

emb = np.array([[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457, -0.49688, -0.17862],
   [0.36808, 0.20834, -0.22319, 0.046283, 0.20098, 0.27515, -0.77127, -0.76804],
   [0.7503, 0.71623, -0.27033, 0.20059, -0.17008, 0.68568, -0.061672, -0.054638],
   [0.042523, -0.21172, 0.044739, -0.19248, 0.26224, 0.0043991, -0.88195, 0.55184],
   [0.17698, 0.065221, 0.28548, -0.4243, 0.7499, -0.14892, -0.66786, 0.11788],
   [-1.1105, 0.94945, -0.17078, 0.93037, -0.2477, -0.70633, -0.8649, -0.56118],
   [0.11626, 0.53897, -0.39514, -0.26027, 0.57706, -0.79198, -0.88374, 0.30119],
   [-0.13531, 0.15485, -0.07309, 0.034013, -0.054457, -0.20541, -0.60086, -0.22407],
   [ 0.13721, -0.295, -0.05916, -0.59235, 0.02301, 0.21884, -0.34254, -0.70213],
   [ 0.61012, 0.33512, -0.53499, 0.36139, -0.39866, 0.70627, -0.18699, -0.77246 ],
   [ -0.29809, 0.28069, 0.087102, 0.54455, 0.70003, 0.44778, -0.72565, 0.62309 ]])

emb.shape

from collections import OrderedDict

# embedding as TF tensor (for now constant; could be tf.Variable() during training)
tf_embedding = tf.constant(emb, dtype=tf.float32)

# input for which we need the embedding
input_str = "like the country"

# build index based on our `vocabulary`
word_to_idx = OrderedDict({w:vocab.index(w) for w in input_str.split() if w in vocab})

# lookup in embedding matrix & return the vectors for the input words
tf.reshape(tf.nn.embedding_lookup(tf_embedding, list(word_to_idx.values())),shape=(-1,3*8))




import tensorflow as tf

v1 = tf.Variable([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
v2 = tf.reshape(v1, [2, 6])
v3 = tf.reshape(v1, [2, 2, -1])
v4 = tf.reshape(v1, [-1])
# v5 = tf.reshape(v1, [2, 4, -1]) will fail, because you can not find such an integer for -1
v6 = tf.reshape(v1, [1, 4, 1, 3, 1])
v6_shape = tf.shape(v6)
v6_squeezed = tf.squeeze(v6)
v6_squeezed_shape = tf.shape(v6_squeezed)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
a, b, c, d, e, f, g = sess.run([v2, v3, v4, v6, v6_shape, v6_squeezed, v6_squeezed_shape])
# print all variables to see what is there
b



embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input).view(2,12)

#x = torch.Tensor([[[1,2,3],[2,2,3],[1,1,1]],[[1,2,3],,[2,2,3]]])

torch.cat((x[0],x[1]),1).size()

x = torch.rand((10,3,4))

x_t = torch.split(x,1,0)
len(x_t)

for x_t in torch.split(x,split_size_or_sections=2,dim=0):
    print(x_t.size())