#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
import pdb

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.e_char = 50
        self.embedding = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=vocab.char_pad)
        self.cnn = CNN(self.e_char, word_embed_size)
        self.highway = Highway(word_embed_size)
        self.dropout = nn.Dropout(0.3)
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        sentence_length, batch_size, _ = input.size()
        x_embed = self.embedding(input)
        x_embed = x_embed.view(sentence_length*batch_size, x_embed.size(2), x_embed.size(3)).permute(0, 2, 1)
        x_convout = self.cnn(x_embed)
        x_highway = self.highway(x_convout)
        x_wordemb = self.dropout(x_highway)
        x_wordemb = x_wordemb.view(sentence_length, batch_size, -1)

        return x_wordemb
        ### END YOUR CODE

