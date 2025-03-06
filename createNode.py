#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from collections import defaultdict


class Node:

    # all these properties are from temporal values
    def __init__(self, current=None, parent=None):
        # initialize node properties
        self.parent = self if None else parent  # None for root

        # a state node has child nodes (state and action pairs))
        self.children = []

        # track N of the current node
        self.N = 0

        # reward comes from the leaf value
        self.R = 0

        # needed?
        self.ucb = np.inf

        # row and column coordinates. (NaN, NaN) for root. Keep track of the current location regardless who you are.
        self.current = None if current is None else current
        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 1          
        | 0 | 1 | 2 | 3 | 4 |   sum(1st col) == 3: depth 2         
        | 1 | 1 | 2 | 3 | 4 |   sum(2nd-5th cols) < 12: depth 2    
        | 2 | 1 | 2 | 3 | 4 |   sum(2nd-5th cols) == 12: depth 3
        |---|---|---|---|---|
        """