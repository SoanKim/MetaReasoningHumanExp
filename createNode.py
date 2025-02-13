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
        self.N = 0
        self.Q = 0

        # track depth
        self.depth = 0
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
        if self.depth == 0:
            return np.sum(self.navi[:, 0]) == 3
        else:
            return any(sum(row) == 5 for row in self.navi)

    def isTerminal(self):
        """
        To check if one node is fully expanded.
        """
        return any(sum(row) == 5 for row in self.navi)

    def UCB(self):
        bestChild = None
        bestUCB = -np.inf
        for child in self.children:
            print("child.N: ", child.N)
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

        while not self.isFullyExpanded():  # or "for i in range(3):"
            if self.N == 0:  # choose an element if the depth is 0
                elementAvail = np.argwhere(tempState[:, 0] == 0).flatten()
                element = random.choice(elementAvail)
            else:
                element = child.current[0]  # cache the stored element if the depth is deeper

            # track available actions(dims)
            tempState[element, 0] = 1

            # store on the invariable state
            self.navi[element, 0] = 1

            # according to the depth
            actionAvail = np.argwhere(tempState[element, :] == 0).flatten()
            print("actionAvail: ", actionAvail)
            action = random.choice(actionAvail)

            # do not record on the invariable state
            tempState[element, action] = 1

            # initialize a child node
            child = Node(prbIdx=None, current=(element, action), parent=self, actionTaken=action)
            print("depth of child:", child.depth)
            print("child's current state: ", child.current)
            leafVal = self.rollout(child)
            self.backprop(leafVal, child)

            # add the child
            self.children.append(child)

        # compare the children at the same depth
        bestChild = self.UCB()
        print("bestChild current: ", bestChild.current)
        self.navi[bestChild.current] = 1
        print("navi:", self.navi)



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
