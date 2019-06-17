#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)        
        self.embed_size = embed_size
        self.dropout=nn.Dropout(0.3)
        
        

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        #x_char_embedding = []
       # for x_padded in input:
        x_padded = input
        x_embedded = self.embeddings(x_padded)
        x_padded_dim = list(x_embedded.size())
#        print(x_embedded.size())
        # need to convert 4d to 3d
        x_embedded = x_embedded.reshape(-1,x_padded_dim[3],x_padded_dim[2])
#        print(x_embedded.size())
        x_conv_out = CNN(in_channel=self.embed_size,out_channel=self.embed_size).forward(x_embedded)
        x_conv_out = torch.squeeze(x_conv_out, -1)
#        print(x_conv_out.size()) 
        x_conv_out = x_conv_out.reshape(x_padded_dim[0],x_padded_dim[1],-1)
#        print(x_conv_out.size()) 
        x_highway = Highway(self.embed_size).forward(x_conv_out)
        #print(x_highway.size())
        x_dout = self.dropout(x_highway)
#        x_char_embedding.append(x_dout)
        #x_char_embedding = torch.stack(x_char_embedding  )
        return x_dout
     
        ### END YOUR CODE

