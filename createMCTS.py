#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from createNode import Node
from copy import deepcopy
from collections import defaultdict
import numpy as np
import random
import math


# class MCTS:
#     """
#     Traverse is horizontal, and expand is vertical.
#     """
#     def __init__(self, prbIdx):
#
#         # initialize game
#         self.prbIdx = prbIdx

# initialize UCB table for selection
ucbTable = np.zeros((3, 5))

# initialize Q table --> going on througout the game
qTable = np.zeros((3, 5))

    # def mcst(self, root):
if __name__ == '__main__':
    for prbIdx in range(3):
        # to check if it's a leaf node or there are available actions
        root = Node(prbIdx=prbIdx)
        root.select(ucbTable=ucbTable)
        reward = root.getReward()
        root.backprop(reward=reward, qTable=qTable, ucbTable=ucbTable)


print("qTable")
print(qTable)