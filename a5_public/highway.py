#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, hidden_dim):
        super(Highway, self).__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_proj = self.relu(self.proj(x))
        x_gate = self.sigmoid(self.gate(x))
        x_highway = x_gate * x_proj + (1-x_gate) * x
        return x_highway

    ### END YOUR CODE


if __name__ == '__main__':
    highway = Highway(64, 0.5).eval()
    a = torch.randn(4, 64)
    b = highway(a)
    print(b.size())
    
    # try gate = 1
    highway.proj.weight.data = torch.zeros_like(highway.proj.weight.data)
    highway.proj.bias.data = torch.zeros_like(highway.proj.bias.data)
    highway.gate.bias.data = torch.ones_like(highway.gate.bias.data) * float('inf')
    b = highway(a)
    print(torch.sum(b))
    assert torch.sum(b) == 0
    
    # try gate = 0
    highway.gate.bias.data = -torch.ones_like(highway.gate.bias.data) * float('inf')
    b = highway(a)
    print((b==a).all())
    assert (b == a).all()
    
    