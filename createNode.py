#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import random
import numpy as np
from copy import deepcopy
from createGame import *
from collections import defaultdict
import math


class Node:
    N = 0
    Q = 0

    def __init__(self, prbIdx=None, current=None, parent=None, actionTaken=None):

        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.navi, self.leafState = self.game.prbInit()

        # row and column coordinates. (-1, -1) for root.
        self.current = current if current is None else (-1, -1)

        # immediate reward
        self.reward = 0

        # initialize node properties
        self.parent = parent  # None for root
        self.actionTaken = actionTaken

        # a state node has child nodes (state and action pairs))
        self.children = defaultdict(tuple)
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
    def isFullyExpanded(self, navi=None):
        if navi is None:
            navi = self.navi
        else:
            navi = navi
            if len(self.children.keys()) < 4:  #correct?
                return np.sum(navi[:, 0]) == 3
            else:
                return any(sum(row) == 5 for row in navi)

    def select(self):
        if not self.isFullyExpanded() or self.game.isTerminal():
            return self
        else:
            maxChild = max(self.children.values(), key=self.children.get)
            # I should update the navi by the maxchild's coordinates
            print(maxChild)
        return maxChild

    def UCB(self, value):
        """
        Fix the problem of division by parent.N, not my N.
        :param value:
        :return:
        """
        if self.parent:
            self.parent = self.parent
        else:
            self.parent = self

        # what if there are multiple children? leafVal should be the mean.
        if self.N > 0:
            if value != 0:
                value = value
            else:
                value = 0
            ucb = value + 2 * math.sqrt(math.log(self.parent.N) / self.N)
            return ucb

    def rollout(self):
        # navi for simulation
        tempState = deepcopy(self.navi)
        while not self.isFullyExpanded(navi=tempState):  # or "for i in range(3):"
            if Node.N == 0:
                elementAvail = np.argwhere(tempState[:, 0] == 0).flatten()
                element = random.choice(elementAvail)
            else:
                if self.game.isTerminal:
                    element = list(self.children.keys())[0]
                else:
                    return self
            self.N += 1
            # Track available actions
            tempState[element, 0] = 1
            actionAvail = np.argwhere(tempState[element, :] == 0).flatten()
            action = random.choice(actionAvail)
            tempState[element, action] = 1
            # initialize a child node
            child = Node(current=(element, action), parent=self, actionTaken=action)

            leafValue = 1/self.contextM[element, action] if self.contextM[element, action] != 0 else 0
            ucb = self.UCB(leafValue)

            # add children
            self.children[(element, action)] = ucb
            self.backprop(reward=leafValue, child=child)

    def backprop(self, reward, child):
        #self.N += 1
        self.Q += reward
        print("self N", self.N)
        print(self.children)

        if child:
            child.N += 1
            child.Q += reward
            print("child.N", child.N)
            print("child.Q", child.Q)
        # if self.parent:
        #     self.parent.N += 1
        #     self.parent.Q += reward
        #     print("parent N", self.parent.N)

    # def expand(self, child):
    #     if child.N > 0:
    #         element = list(self.children.keys())[0]
    #     else:
    #         element = None
    #
    #     # If never visited, don't expand.
    #     if Node.N == 0:

    #     actionAvail = self.game.legalMove(element=element)
    #     action = random.choice(actionAvail)
    #     value = self.simulate(action)
    # else:
    #     return self.children[action] = int(np.nan)



        # else:
        #     tempEnv = deepcopy(node.state)
        #     positions = []
        #     values = []
        #     while np.sum(node.state[:, 0]) < 3:
        #         # row = actions in depth 1
        #         rowCandi = np.argwhere(tempEnv[:, 0] == 0).flatten()
        #         row = random.choice(rowCandi)
        #
        #         # col = actions in depth 2
        #         colCandi = np.argwhere(tempEnv[row, :] == 0).flatten()
        #         colCandi = [c for c in colCandi]
        #         col = random.choice(colCandi)
        #
        #         # mark elem choice
        #         self.navi[row, 0] = 1
        #
        #         # rollout info is volatile
        #         tempEnv[row, 0] = 1
        #         tempEnv[row, col] = 1
        #         positions.append((row, col))
        #
        #         # value of those actions
        #         val = self.contextM[row, col]
        #         if val != 0:
        #             values.append(np.round(1 / val, 2))
        #         else:
        #             values.append(0)
        #
        #         # Calculate UCB
        #         ucb = self.UCB(node=node, leafVal=val)
        #         print("ucb: {}".format(ucb))
        #
        #     valueIdx = values.index(max(values))
        #     bestElem = positions[valueIdx][0]  # row
        #
        #     # update children
        #     for i, (row, col) in enumerate(positions):
        #         self.root.children[row] = values[i]
        #
        #     return bestElem




    #     """ Updates reward and visit counts, propagating up the tree. """
    #     if self.parent is None:
    #         # increment the visit of the current node's
    #         self.visitSum += 1
    #         # increment the value of current node's
    #         node.valueSum = self.UCB(node, val)
    #         # Calculate the new average reward
    #         node.valueSum += self.UCB(node, val)
    #
    #     else:
    #         # increment the visit of the parent's
    #         node.parent.visitSum += 1
    #         node.parent.valueSum += np.sum(node.valueSum) / node.parent.visitSum
    #         # increment the visit of the current node's
    #         node.visitSum += 1
    #         # increment the value of current node's
    #         ucb = self.UCB(node, val)
    #         # Calculate the new average reward
    #         node.valueSum += ucb
    #
    #
    #
    # def trackDepth(self):
    #     if np.sum(self.current[:, 0]) < 3:
    #         self.depth = 0
    #     elif np.sum(self.current[:, 0]) == 3 and np.sum(self.current[:, 1:]) < 12:
    #         self.depth = 1
    #     elif np.sum(self.current[:, 0]) == 3 and np.sum(self.current[:, 1:]) == 12:
    #         self.depth = 2
    #     return self.depth





# if __name__ == '__main__':
#
#     game = Game()
#     contextM, cardAvail, answer, navi, leafState = game.prbInit()
#     root = Node()
#     print(root.parent)