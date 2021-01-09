#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
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
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        input_embeddings = self.decoderCharEmb(input) # (length, batch_size, char_embedding_size)
        scores, dec_hidden = self.charDecoder(input_embeddings, dec_hidden) # (length, batch_size, hidden_size)
        scores = self.char_output_projection(scores) # (length, batch_size, vocab_size)
        
        return scores, dec_hidden
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        
        # 1. build the input (len, b_size) and target mask
        input_sequence = char_sequence[:-1, :]
        target_sequence = char_sequence[1:, :] # (length, batch_size)
        target_masks = (char_sequence[1:, :] != self.target_vocab.char_pad).float() # (length, batch_size)
        
        # 2. pass input to CharDecoderLSTM and get a score vector, then take softmax to get pt
        scores, dec_hidden = self.forward(input_sequence, dec_hidden)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_char = criterion(scores.permute(1, 2, 0), target_sequence.permute(1, 0)) # (N,C,d) and (N,d)
        loss_char = torch.sum(loss_char.permute(1, 0)  * target_masks)
#         P = F.log_softmax(scores, dim=-1) # (length, batch_size, vocab_size)
        
        # 3. calcualte cross entropy loss
#         loss_char = torch.gather(P, dim=-1, index=target_sequence.unsqueeze(-1)).squeeze() * target_masks
#         loss_char = -torch.sum(loss_char)
        
        return loss_char
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        output_word =  ['' for i in range(batch_size)]
        current_char = torch.ones((1, batch_size), dtype=torch.long, device=device) * self.target_vocab.start_of_word
        for t in range(max_length):
            input_embeddings = self.decoderCharEmb(current_char) # (length, batch_size, char_embedding_size)
            scores, initialStates = self.charDecoder(input_embeddings, initialStates) # (length, batch_size, hidden_size)
            scores = self.char_output_projection(scores) # (length, batch_size, vocab_size)
            P = self.softmax(scores)
            current_char = torch.argmax(P, dim=-1, keepdim=False)
            output_word = [output_word[i]+self.target_vocab.id2char[current_char[0, i].item()] for i in range(batch_size)]
        output_word = [output_word[i].split('}')[0] for i in range(batch_size)]
        return output_word
        ### END YOUR CODE

