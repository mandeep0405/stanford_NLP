#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.target_vocab = target_vocab
        self.char_output_projection = nn.Linear(hidden_size,len(target_vocab.char2id),bias=True)
        pad_token_idx = target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id),char_embedding_size,padding_idx=pad_token_idx)

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. 
        A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)
#        X = pack_padded_sequence(X, source_lengths)

#            print(x_t.size())
        output,dec_hidden = self.charDecoder(X,dec_hidden)
        scores = self.char_output_projection(output)

        return (scores,dec_hidden)
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). 
        Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, 
        obtained from the output of the word-level decoder. 
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, 
        computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} 
        ###         from the handout (e.g., <START>,m,u,s,i,c,<END>).
        char_sequence = char_sequence[:-1]
        
        input = char_sequence[:-1]
        
        target_padded = char_sequence[1:]        
        
        scores,dec_hidden = self.forward(input,dec_hidden)
        
        P = F.log_softmax(scores, dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.target_vocab['<pad>']).float()

        # Compute log probability of generating true target chars
        target_chars_log_prob = torch.gather(P, index=target_padded.unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_chars_log_prob.sum() # mhahn2 Small modification from A4 code.

        return scores
        
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, 
        a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, 
        each of which has length <= max_length.
        The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        decodedWords = []
        batch_size = initialStates[0].shape[1]
        current_char = torch.tensor([self.target_vocab.start_of_word for _ in range(batch_size)], device=device).unsqueeze(0)
        lstmStates = initialStates
#        print(current_char.size()) (1,b)
        decodedIndices = current_char.permute(1,0)
        for t in range(max_length):
            scores,lstmStates = self.forward(current_char,lstmStates)
            P = F.log_softmax(scores, dim=-1) # (1, batch, self.vocab_size)
#            print(P.size())
            current_char = torch.argmax(P,dim=2) # (1,b,1)
            current_charIdx = current_char.unsqueeze(2).squeeze(0)
#            print(current_char.size())
            decodedIndices = torch.cat((decodedIndices,current_charIdx),dim=1)
            
        decodedWords = decodedIndices.tolist()
        # remove start index and chars after end
        decodedWords = [i[1:] for i in decodedWords] # remove start
#        print(len(decodedWords[0]))
        output = []
        for words in decodedWords:
            chars = []
            for idx in words:
                if idx == self.target_vocab.end_of_word :
                    break
                chars.append(self.target_vocab.id2char[idx])
            output.append(chars)

        return output
        ### END YOUR CODE

