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

"""
TO DO:
keep track of every child's visits
update qtable and ucb table of every child
"""
class Node:
    visits = defaultdict(lambda: 0)  # please initialize this every trial
    qTable = np.zeros((3, 5))
    ucbTable = np.empty((3, 5))
    probTable = np.empty((3, 5))  # prob of successful visits

    # all these properties are from temporal values
    def __init__(self, prbIdx=None, current=None, parent=None):
        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.monitor, self.leafState = self.game.prbInit()
        print("self.leafstate", self.leafState)
        # C for UCB
        self.exploreConstant = 2
        self.gamma = 0.95

        # row and column coordinates. (-1, -1) for root. Keep track of the current location regardless who you are.
        self.current = (-1, -1) if current is None else current

        # immediate reward
        self.reward = 0

        # initialize node properties
        self.parent = self if None else parent  # None for root

        # a state node has child nodes (state and action pairs))
        self.children = []  # {}  # defaultdict(lambda: 0)  or [[], []]  or defaultdict(tuple) for self.children[(element, action)] = leafValue

        # track depth
        self.depth = 0

        # track N of the current node
        self.N = 0

        # reward comes from the leaf value
        self.R = 0

        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 0          before selecting a leaf val: depth 2
        | 0 | 0 | 1 | 2 | 3 |   sum(1st col) == 3: depth 1         after selecting a leaf val: depth 3
        | 1 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) < 12: depth 1    after receiving reward = terminal
        | 2 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) == 12: depth 2
        |---|---|---|---|---|
        """

    def isFullyExpanded(self):  # checking if it reached the leaf state
        rowSum = [sum(row) for row in self.monitor]

        if np.sum(self.monitor[:, 0]) < 3 and 5 not in rowSum:
            self.depth = 0  # num of children = 3
            return False, False
        elif np.sum(self.monitor[:, 0]) == 3 and 5 not in rowSum:
            self.depth = 1  # num of children = 4
            return True, False
        elif np.sum(self.monitor[:, 0]) == 3 and 5 in rowSum:  # any(li) means all elements of list are True
            self.depth = 2  # num of cards varies across trials
            return True, True

    def search(self):
        if self.N == 0:
            while self.isFullyExpanded() != (True, True):
                self.select()  # get actions recursively
        else:
            self.select()
        parent = Node(prbIdx=self.prbIdx, current=(self.current[0], 0), parent=self)
        newChild = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
        self.children.append(newChild)

        # add children

        print("len children: {}".format(len(self.children)))
        print("self.current:{}".format(self.current))

        if self.isFullyExpanded() == (True, True):
            final = True
        else:
            final = False
        rwd = self.rollout(final=final)
        self.backprop(rwd)
        print("rwd: {}".format(rwd))
        print("Node.visits", Node.visits)

    def getReward(self):
        if self.depth == 2:
            finalAction = self.current[0] * 4 + self.current[1] - 1
        else:
            raise ValueError('Not yet a leaf node (depth!= 2)')
        return self.leafState[finalAction]

    def backprop(self, reward):
        for child_i, child in enumerate(self.children):
            print("child #: {}, current: {}".format(child_i, child.current))
            Node.visits[child.current] += 1
            self.updateUCB(reward=reward)
            self.N += 1
            self.R += reward

            if child.parent is not None:
                Node.visits[child.parent.current] += 1

    def rollout(self, final: bool, rolloutPolicy=None):
        """
        todo: get the reward from the leaf node
        :input: s', a' d
        :return: R from depth 2
        """

        if self.depth == 2:
            numCards = self.contextM[self.current]
            if numCards == 0:
                r = 0
            else:
                r = 1/numCards
            if final:
                rwd = self.getReward()
                R = r + r * pow(self.gamma, self.depth) * rwd
            else:
                R = r + r * pow(self.gamma, self.depth)
        else:
            raise ValueError('Not yet a leaf node (depth != 2)')
        return R

    def select(self, treePolicy=None):
        """
        todo: choose actions recursively till the leaf
        input: current state
        :return: next state and depth + 1
        """
        if self.depth == 0:
            maxUCB = max(Node.ucbTable[:, 0])
            actions = np.argwhere(Node.ucbTable[:, 0] == maxUCB).flatten()

            # update the location
            if len(actions) > 1:
                action = random.choice(actions)
            else:
                action = actions
            print("maxUCB", maxUCB)
            print("best action:", action)

            self.current = (action, 0)

        elif self.depth == 1:
            maxUCB = max(Node.ucbTable[self.current[0], 1:])
            actions = np.argwhere(Node.ucbTable[self.current[0], 1:] == maxUCB).flatten() + 1
            # update the location
            if len(actions) > 1:
                action = random.choice(actions)
            else:
                action = actions
            print("maxUCB", maxUCB)
            print("best action:", action)
            self.current = (self.current[0], action)

        else:
            self.current = self.current
            print("It's already at leaf node. Roll out.")

        self.monitor[self.current] = 1
        print("self.monitor")
        print(self.monitor)
        print("depth: {}".format(self.depth))

    def updateUCB(self, reward):
        childN = Node.visits[self.current]
        if childN == 0:
            Node.ucbTable[self.current] = np.inf
        if self.depth == 0:
            parentN = Node.visits[-1, -1]
        elif self.depth == 1:
            parentN = Node.visits[self.current[0], 0]
        else:
            parentN = Node.visits[self.current]
        print("parentN: {}, childN: {}".format(parentN, childN))
        # delta = reward - Node.qTable[self.current] / childN

        Node.qTable[self.current] = Node.qTable[self.current] + reward
        # update the root as well
        Node.qTable[self.current[0], 0] = Node.qTable[self.current[0], 0] + reward

        ucb = Node.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parentN) / childN)
        print("ucb:{}".format(ucb))
        Node.ucbTable[self.current] = Node.ucbTable[self.current] + ucb
        # update the root as well
        Node.ucbTable[self.current[0], 0] = Node.ucbTable[self.current[0], 0] + ucb
        # print("self.contextM", self.contextM)
        print("Node.ucbTable", Node.ucbTable)
        print("Node.qTable", Node.qTable)