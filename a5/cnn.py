#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, in_channel: int=None,out_channel: int=None,k: int=5,m_word : int=21):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()
        self.conv = torch.nn.Conv1d(in_channel, out_channel,k)
        self.maxpool = torch.nn.MaxPool1d(m_word-k+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param X (Tensor): Tensor of char-level embedding with shape ( 
                                    batch_size, e_char, m_word), where e_char = embed_size of char, 
                                    m_word = max_word_length.
        @return X_conv_out (Tensor): Tensor of word-level embedding with shape (max_sentence_length,
                                    batch_size,embed_size)

        """
        
        x_conv = self.conv(x)
        x_conv_out = self.maxpool(F.relu(x_conv))
        return x_conv_out 
### END YOUR CODE

