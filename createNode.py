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
    visits = defaultdict(lambda: 0)  # please initialize this every trial

    def __init__(self, prbIdx=None, current=None, parent=None):
        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.monitor, self.leafState = self.game.prbInit()

        # C for UCB
        self.exploreConstant = 2

        # row and column coordinates. (-1, -1) for root. Keep track of the current location regardless who you are.
        self.current = (-1, -1) if current is None else current

        # immediate reward
        self.reward = 0

        # initialize node properties
        self.parent = parent  # None for root

        # a state node has child nodes (state and action pairs))
        self.children = defaultdict(lambda: 0) #[[], []]
        # self.children = defaultdict(tuple) --> for self.children[(element, action)] = leafValue

        # tracking visits and rewards (parent N only?)
        self.N = 0

        # track depth
        self.depth = 0

        # initialize UCB table for selection
        self.ucbTable = np.full((3, 5), np.inf)

        # initialize Q table
        self.qTable = np.zeros((3, 5))

        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 0          before selecting a leaf val: depth 2
        | 0 | 0 | 1 | 2 | 3 |   sum(1st col) == 3: depth 1         after selecting a leaf val: depth 3
        | 1 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) < 12: depth 1    after receiving reward = terminal
        | 2 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) == 12: depth 2
        |---|---|---|---|---|
        """

    def isFullyExpanded(self, state, traverse: bool):  # checking if it reached the leaf state
        rowSum = [sum(row) for row in state]
        if traverse:
            d1idx = 3
            d2idx = 5
        else:
            d1idx = 1
            d2idx = 2

        # modify the number of actions as you like.
        if np.sum(state[:, 0]) < d1idx and d2idx not in rowSum:
            self.depth = 0  # num of children = 3
            return False, False
        elif np.sum(state[:, 0]) == d1idx and d2idx not in rowSum:
            self.depth = 1  # num of children = 4
            return True, False
        elif np.sum(state[:, 0]) == d1idx and d2idx in rowSum:  # any(li) means all elements of list are True
            self.depth = 2  # num of cards varies across trials
            return True, True

    def updateQtable(self, reward):
        # Update Q by depth
        for child in self.children:  # removing depth
            childN = Node.visits[self.current]
            if self.depth == 0:
                parentN = Node.visits[-1, -1]
            elif self.depth == 1:
                parentN = Node.visits[self.current[0], 0]
            else:
                parentN = Node.visits[self.current]
            print("parentN: {}, childN: {}".format(parentN, childN))
            delta = reward - self.qTable[self.current] / childN
            self.qTable[self.current] = self.qTable[self.current] + delta
            ucb = self.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parentN) / childN)
            self.ucbTable[self.current] = self.ucbTable[self.current] + ucb

    def getBestArm(self):
        best1dChild = np.argmax(self.ucbTable[:, 0])
        best2dChild = np.argmax(self.ucbTable[self.current[0], :])
        return best1dChild, best2dChild


    # def getQvalUCB(self, actionAvail, parent):  # This is the ultimate source of problem
    #     bestAction = None
    #     bestUCB = 0
    #     # print("++++++++++++++ depth: {} ++++++++++++".format(self.depth)) # works correctly.
    #     #for act in actionAvail:  # This is the cause!
    #
    #         if self.depth == 0:
    #             self.current = (act, 0)
    #         else:
    #             self.current = (self.current[0], act)
    #
    #         if self.contextM[self.current] != 0:
    #             leafVal = 1/self.contextM[self.current]
    #         else:
    #             leafVal = 0
    #
    #         Node.visits[self.current] += 1
    #
    #         child = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
    #         self.backprop(child=child, value=leafVal)
    #         self.updateQtable(child=child, value=leafVal)
    #
    #         if child.N == 0:
    #             ucb = np.inf
    #         else:
    #             ucb = self.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parent.N) / child.N)
    #         self.ucbTable[self.current] = self.ucbTable[self.current] + ucb
    #
    #         if ucb != np.inf and ucb > bestUCB:
    #             bestUCB = ucb
    #             bestAction = act
    #     print("Node.visits: {}".format(Node.visits))
    #
    #     if bestAction is None:
    #         bestAction = random.choice(actionAvail)
    #     # print("######### QTable ##########")
    #     # print(self.qTable)
    #     # print("+++++++++ UCB table +++++++++")
    #     # print(self.ucbTable)
    #
    #     return bestAction, bestUCB

    def chooseAction(self, heuristic: bool, state):  # select an action for the next state
        if self.depth == 0:
            actionAvail = np.argwhere(state[:, 0] == 0).flatten()  # element
            if not heuristic:
                action = random.choice(actionAvail)
            else:
                maxQ = max(self.qTable[:, 0])
                actionCandi = np.argwhere(self.qTable[:, 0] == maxQ).flatten()
                action = random.choice(np.intersect1d(actionAvail, actionCandi))

            self.current = (action, 0)
            # keep track of the action on temporary monitor
            state[action, 0] = 1
            # keep track of the action on the real monitor
            # self.monitor[action, 0] = 1
            print("================ element's actionAvail: {} =================".format(actionAvail))

        else:
            actionAvail = np.argwhere(state[self.current[0], :] == 0).flatten()  # action
            print("+++++++++++++++ action's actionAvail: {} ++++++++++++++++++".format(actionAvail))
            if not heuristic:
                action = random.choice(actionAvail)
            else:
                maxQ = max(self.qTable[self.current[0], :])
                actionCandi = np.argwhere(self.qTable[self.current[0], :] == maxQ).flatten()
                action = random.choice(np.intersect1d(actionAvail, actionCandi))

            self.current = (self.current[0], action)

            # keep track of the action on temporary monitor
            state[self.current[0], action] = 1
            # keep track of the action on the real monitor
            # self.monitor[self.current[0], action] = 1

        return action

    def select(self):
        """
        This is to visit each node only once.
        """
        state = np.zeros((3, 5))
        while self.isFullyExpanded(state=state, traverse=True) == (False, False):
            element = self.chooseAction(heuristic=True, state=state)
            # Expand
            self.children[element, 0] = 0  # self.children[0].append(child)
            print("chosen element: ", element)

            while self.isFullyExpanded(state=state, traverse=True) == (True, False):
                action = self.chooseAction(heuristic=True, state=state)
                # Expand
                self.children[self.current[0], action] = 0  # self.children[1].append(child)
                print("chosen action: ", action)

    def rollout(self):
        if self.contextM[self.current] != 0:
            leafValue = 1 / self.contextM[self.current]
        else:
            leafValue = 0
            # different Q values per depth
        if self.isFullyExpanded(state=self.monitor, traverse=True) == (True, True):  # if terminal:
            rwd = self.leafState[self.current[0]*4+self.current[1]]
            reward = leafValue * rwd
        else:  # if not terminal, just a leaf value
            reward = leafValue

        self.backprop(reward=reward)
        self.updateQtable(reward=reward)

    def backprop(self, reward):
        if self.isFullyExpanded(state=self.monitor, traverse=True) == (True, True):
            for child in self.children:  # Is it better to divide the depth?
                Node.visits[self.current] += 1
                self.children[self.current] += reward

                if child.parent is not None:
                    child.parent.N += 1
