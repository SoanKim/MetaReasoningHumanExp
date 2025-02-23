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

# TO DO: keep track of every child's visits & update qtable and ucb table of every child


class Node:
    visits = defaultdict(lambda: 0)  # please initialize this every trial
    qTable = np.zeros((3, 5))
    ucbTable = np.empty((3, 5))
    # probTable = np.empty((3, 5))  # prob of successful visits

    # all these properties are from temporal values
    def __init__(self, prbIdx=None, current=None, parent=None):
        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.monitor, self.leafState = self.game.prbInit()
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

    # THIS IS WRONG
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

    def search(self, parent):
        # Consider depth
        if self.N == 0:
            rwd = self.rollout(final=False, parent=parent)
            self.backprop(rwd)
            print("reward when N is 0", rwd)
        else:  # THIS LOOP IS NOT REACHED
            while self.depth < 2:
                self.select(parent=parent)
                rwd = self.rollout(final=True, parent=parent)
                print("reward when N is not 0 and depth < 2", rwd)
                self.backprop(rwd)
            reward = self.getReward()
            print("final reward", reward)
            self.backprop(reward=reward)

    def getReward(self):
        if self.depth == 2:
            finalAction = self.current[0] * 4 + self.current[1] - 1
        else:
            raise ValueError('Not yet a leaf node (depth!= 2)')
        return self.leafState[finalAction]

    def backprop(self, reward):
        for child_i, child in enumerate(self.children):
            Node.visits[child.current] += 1
            self.updateUCB(reward=reward)
            self.N += 1
            self.R += reward

            if child.parent is not None:
                Node.visits[child.parent.current] += 1

    def rollout(self, parent, final: bool, rolloutPolicy=None):
        """
        todo: choose actions recursively to get the reward from the leaf node
        :input: s', a' d
        :return: R from depth 2
        """
        while self.depth < 2:
            self.select(parent=parent)
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
        return R

    def select(self, parent, treePolicy=None):
        """
        todo: choose one action among current states
        input: current state
        :return: next state and depth + 1
        """
        print("currentDepth: ", self.depth)
        if self.depth == 0:
            monitorRange = self.monitor[:, 0]
            ucbRange = Node.ucbTable[:, 0]
        elif self.depth == 1:
            monitorRange = self.monitor[self.current[0], :]
            ucbRange = Node.ucbTable[self.current[0], :]
        else:
            monitorRange = None
            ucbRange = None
        print("UCB range:", ucbRange)
        print("depth before: {}".format(self.depth))
        legalAction = np.argwhere(monitorRange == 0).flatten()  # rows
        print("+++++++++++ legalAction: +++++++++++++", legalAction)
        legalUCBmat = np.take(ucbRange, legalAction)
        print("%%%%%%%%%%%% legalUCBmat: %%%%%%%%%%%%", legalUCBmat)
        maxUCB = max(legalUCBmat)
        print("maxUCB", maxUCB)
        actions = np.argwhere(legalUCBmat == maxUCB).flatten()

        # update the location
        if len(actions) > 1:
            action = random.choice(actions)
        else:
            action = actions[0]
        print("best action:", action)

        if self.depth == 0:
            self.current = (action, 0)
        elif self.depth == 1:
            self.current = (self.current[0], action)
        else:
            self.current = None
        self.depth += 1

        self.monitor[self.current] = 1
        Node.visits[self.current] += 1
        print("self.monitor")
        print(self.monitor)
        print("depth after: {}".format(self.depth))
        newChild = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
        self.children.append(newChild)

    def updateUCB(self, reward): #parent's current?
        childN = Node.visits[self.current]
        if childN == 0:
            Node.ucbTable[self.current] = np.inf
        else:
            Node.ucbTable[self.current] = reward  # PROBLEMP: UCB table's node N is too high and it's not exploring
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