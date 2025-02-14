#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import random
import numpy as np
from copy import copy, deepcopy
from createGame import *
from collections import defaultdict
import math


class Node:

    def __init__(self, prbIdx=None, current=None, parent=None, actionTaken=None):

        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.navi, self.leafState = self.game.prbInit()

        # C for UCB
        self.exploreConstant = 2

        # row and column coordinates. (-1, -1) for root.
        self.current = (-1, -1) if current is None else current

        # immediate reward
        self.reward = 0

        # initialize node properties
        self.parent = parent  # None for root
        self.actionTaken = actionTaken

        # a state node has child nodes (state and action pairs))
        self.children = []
        # self.children = defaultdict(tuple) --> for self.children[(element, action)] = leafValue

        # tracking visits and rewards
        self.N = defaultdict()
        self.Q = defaultdict(lambda:0)

        # track depth
        self.depth = 0

        # initialize UCB table for selection
        self.ucbTable = np.full((3, 5), np.inf)

        """
        |---|---|---|---|---|
        | 0 | 0 | 1 | 2 | 3 |
        | 1 | 0 | 1 | 2 | 3 |
        | 2 | 0 | 1 | 2 | 3 |
        |---|---|---|---|---|
        sum(1st col) < 3: depth 0
        sum(1st col) == 3: depth 1
        sum(2nd-4th cols) < 12: depth 1
        sum(2nd-4th cols) == 12: depth 2
        
        before selecting a leaf val: depth 2
        after selecting a leaf val: depth 3
        after receiving reward = terminal
        """

    def isFullyExpanded(self):
        """
        Fully expanded = leaf state
        """
        # if self.depth == 0:
        #     return np.sum(self.navi[:, 0]) == 3
        # else:
        return any(sum(row) == 5 for row in self.navi)

    def UCB(self):
        bestChild = None
        bestUCB = -np.inf
        for child in self.children:

            ucb = child.Q + self.exploreConstant * math.sqrt(math.log(child.parent.N) / child.N)
            # update N and Q of the parent and the child(self)
            if ucb > bestUCB:
                bestChild = child
                bestUCB = ucb
        self.navi[bestChild.current[0], bestChild.current[1]] = 1

        return bestChild

    def select(self):  # select a child among children
        # navi is a state for simulation
        tempState = deepcopy(self.navi)

        if np.sum(tempState[:, 0]) > 3:  # root
            actionAvail = np.argwhere(tempState[self.current[0], :] == 0).flatten()
        else:
            actionAvail = np.argwhere(tempState[:, 0] == 0).flatten() # root

        for action in actionAvail:
            if np.sum(tempState[:, 0]) < 3:  # root
                element = action
                action = 0
                self.navi[element, 0] = 1
            else:
                element = self.current[0]

            self.current = (element, action)

            # track element
            tempState[self.current] = 1
            self.N[self.current] = 1
            print("self.Q[self.current]", self.Q[self.current])
            print("self.N[self.current]", self.N[self.current])

            self.ucbTable[element, action] = \
                self.Q[self.current] + self.exploreConstant * math.sqrt(math.log(self.N[self.current]) / self.N[self.current])

        #print(self.ucbTable)




        #     leafVal = self.rollout(child)
        #     self.backprop(leafVal, child)
        #
        #     # add the child
        #     self.children.append(child)
        #
        # # compare the children at the same depth
        # bestChild = self.UCB()
        # print("bestChild current: ", bestChild.current)
        # self.navi[bestChild.current] = 1
        # print("navi:", self.navi)



    def expand(self, bestChild):
        """
        expand - rollout - backprop
        """
        # store the bestChild's coordinate on the invariable state

        element = bestChild.current[0]
        actionAvail = np.argwhere(self.navi[element, :] == 0).flatten()
        action = random.choice(actionAvail)
        newNode = Node(current=(element, action), parent=bestChild, actionTaken=action)

        bestChild.depth += 1

        return newNode

    def rollout(self, node):
        """
        roll out function only runs when the root is not fully expanded under MCTS class.
        return: leaf value
        """
        # recall the stored element
        element = node.current[0]

        # choose a random action
        actionAvail = np.argwhere(self.navi[element, :] == 0).flatten()
        action = random.choice(actionAvail)

        # leaf value
        leafValue = 1 / self.contextM[element, action] if self.contextM[element, action] != 0 else 0

        # update the depth after expansion
        return leafValue

    def backprop(self, value, node):
        print("================================")
        # update the parent
        node.parent.Q += value
        node.parent.N += 1

        print("child.parent.N", node.parent.N)
        print("child.parent.Q", node.parent.Q)

        # update the child
        node.N += 1
        node.Q += value
        print("child.N", node.N)
        print("child.Q", node.Q)

        print("node depth:", self.depth)


# if __name__ == '__main__':
#
#     game = Game()
#     contextM, cardAvail, answer, navi, leafState = game.prbInit()
#     root = Node()
#     print(root.parent)
