#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 5, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        x_convout = self.pool(self.relu(self.conv(x)))
        x_convout = x_convout.squeeze(-1)
        
        return x_convout

    ### END YOUR CODE

