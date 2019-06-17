#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn.functional as F

class Highway(torch.nn.Module):
    def __init__(self, embed_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Highway, self).__init__()
        self.proj = torch.nn.Linear(embed_size, embed_size)
        self.gate = torch.nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        param x: tensor with shape of [batch_size, embed_size]
        :return: tensor with shape of [max_sentence_length,batch_size, embed_size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """
        gate = torch.sigmoid(self.gate(x))
        nonlinear = F.relu(self.proj(x))
        x_highway = torch.mul(nonlinear,gate) + torch.mul(x,(1 - gate))
        
        return x_highway

### END YOUR CODE 

