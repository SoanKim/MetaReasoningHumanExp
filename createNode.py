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
    visits = defaultdict(lambda: 0)

    def __init__(self, prbIdx=None, current=None, parent=None):
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

        # a state node has child nodes (state and action pairs))
        self.children = []
        # self.children = defaultdict(tuple) --> for self.children[(element, action)] = leafValue

        # tracking visits and rewards
        self.N = 0
        self.Q = 0

        # track depth
        self.depth = 0

        # initialize UCB table for selection
        self.ucbTable = np.zeros((3, 5))

        # initialize Q table
        self.qTable = np.zeros((3, 5))

        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 0          before selecting a leaf val: depth 2
        | 0 | 0 | 1 | 2 | 3 |   sum(1st col) == 3: depth 1         after selecting a leaf val: depth 3
        | 1 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) < 12: depth 1    after receiving reward = terminal
        | 2 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) == 12: depth 2
        |---|---|---|---|---|
        """

    def isFullyExpanded(self, state=None):  # checking if it reached the leaf state
        if state is None:
            state = self.navi
        else:
            state = state
        return np.sum(state[:, 0]) == 3, any(sum(row) == 5 for row in state)

    def updateQtable(self, child, value, alpha=0.1):
        delta = (1 / (Node.visits[child.current])) * (value - self.qTable[child.current])
        self.qTable[child.current] = self.qTable[child.current] + alpha * delta

    def getQvalUCB(self, actionAvail, parent):
        bestAction = None
        bestUCB = 0
        for act in actionAvail:
            if self.depth == 0:
                self.current = (act, 0)
            else:
                self.current = (self.current[0], act)

            if self.contextM[self.current] != 0:
                leafVal = 1/self.contextM[self.current]
            else:
                leafVal = 0

            Node.visits[self.current] += 1
            child = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
            self.backprop(child=child, value=leafVal)
            self.updateQtable(child=child, value=leafVal)

            if child.N == 0:
                ucb = np.inf
            else:
                ucb = self.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parent.N) / child.N)
            self.ucbTable[self.current] = self.ucbTable[self.current] + ucb

            if ucb != np.inf and ucb > bestUCB:
                bestUCB = ucb
                bestAction = act

        if bestAction is None:
            bestAction = random.choice(actionAvail)
        print("######### QTable ##########")
        print(self.qTable)
        print("+++++++++ UCB table +++++++++")
        print(self.ucbTable)

        return bestAction, bestUCB

    def select(self, heuristic: bool, parent, state):  # select an action for the next state
        if self.depth == 0:
            actionAvail = np.argwhere(state[:, 0] == 0).flatten()
        else:
            actionAvail = np.argwhere(state[self.current[0], :] == 0).flatten()

        if not heuristic:
            action = random.choice(actionAvail)
        else:
            action, _ = self.getQvalUCB(actionAvail, parent)

        if self.depth == 0:
            self.current = (action, 0)
        else:
            self.current = (self.current[0], action)
        self.navi[self.current] = 1

        return action

    def expand(self, parent):
        tempState = deepcopy(self.navi)
        while not self.isFullyExpanded(state=tempState) == (True, True):
            element = self.select(heuristic=True, parent=parent, state=tempState)
            self.current = (element, 0)
            tempState[self.current] = 1

            while self.isFullyExpanded(state=tempState) == (True, False):
                self.depth = 1
                action = self.select(heuristic=True, parent=parent, state=tempState)
                self.current = (element, action)
                tempState[self.current] = 1

    def rollout(self, child):
        leafVal = 1/self.contextM[child.current]
        # update the counts of the qTable
        self.updateQtable(child=child, value=leafVal)
        return leafVal

    def backprop(self, child, value):
        # update the child
        child.N += 1
        child.Q += value

        child.parent.N += 1
        child.parent.Q += value

        Node.visits[child.current] = Node.visits[child.current] + 1

