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
    qTable = np.zeros((3, 5))
    ucbTable = np.empty((3, 5))
    probTable = np.empty((3, 5))  # prob of successful visits

    # all these properties are from temporal values
    def __init__(self, prbIdx=None, current=None, parent=None):
        # initialize game
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.monitor, self.leafState = self.game.prbInit()

        # C for UCB
        self.exploreConstant = 2
        self.gamma = 1

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
        self.N = defaultdict(tuple)

        # reward comes from the leaf value
        self.R = defaultdict(tuple)

        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 0          before selecting a leaf val: depth 2
        | 0 | 0 | 1 | 2 | 3 |   sum(1st col) == 3: depth 1         after selecting a leaf val: depth 3
        | 1 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) < 12: depth 1    after receiving reward = terminal
        | 2 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) == 12: depth 2
        |---|---|---|---|---|
        """

    def isFullyExpanded(self, state=None):  # checking if it reached the leaf state
        if not state:
            state = self.monitor
        else:
            state = state
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
        # print("----------- context ------------", self.contextM)
        childN = Node.visits[self.current]
        if self.depth == 0:
            parentN = Node.visits[-1, -1]
        elif self.depth == 1:
            parentN = Node.visits[self.current[0], 0]
        else:
            parentN = Node.visits[self.current]
        print("parentN: {}, childN: {}".format(parentN, childN))
        delta = reward - Node.qTable[self.current] / childN
        print("delta:{}".format(delta))
        Node.qTable[self.current] = Node.qTable[self.current] + delta
        # update the root as well
        Node.qTable[self.current[0], 0] = Node.qTable[self.current[0], 0] + delta

        ucb = Node.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parentN) / childN)
        Node.ucbTable[self.current] = Node.ucbTable[self.current] + ucb
        # update the root as well
        Node.ucbTable[self.current[0], 0] = Node.ucbTable[self.current[0], 0] + ucb
        # print("self.contextM", self.contextM)
        # print("Node.ucbTable", Node.ucbTable)

    def chooseAction(self):  # select an action for the next state
        """
        input: state
        :return: R
        """

        maxQ = max(Node.qTable[:, 0])
        actionCandi = np.argwhere(Node.qTable[:, 0] == maxQ).flatten()
        print(" %%%%%%%%%%%%%%% PROBLEM %%%%%%%%%%%% ")
        print(actionCandi)
        action = random.choice(np.intersect1d(nextActions, actionCandi))

        # Update the current state
        self.current = (action, 0)

        # keep track of the action on temporary monitor
        self.monitor[action, 0] = 1
        # keep track of the action on the real monitor
        # self.monitor[action, 0] = 1

        else:
            maxQ = max(Node.qTable[self.current[0], :])
            actionCandi = np.argwhere(Node.qTable[self.current[0], :] == maxQ).flatten()
            action = random.choice(np.intersect1d(nextActions, actionCandi))

            # Update the current state
            self.current = (self.current[0], action)
            # keep track of the action on temporary monitor
            self.monitor[self.current[0], action] = 1
            # keep track of the action on the real monitor
            # self.monitor[self.current[0], action] = 1
        return action

    def select(self):  # don't return because it needs to run continuously
        """
        :input: state
        :return: a(argmax Q(<s, a>))
        """
        while self.isFullyExpanded() == (False, False):
            if self.N == 0:
                R = self.rollout()
            else:
                if self.depth == 0:
                    actions = np.argwhere(Node.qTable[:, 0]).flatten()
                else:
                    actions = np.argwhere(Node.qTable[self.current[0], :]).flatten()
                ucb = self.exploreConstant * math.sqrt(math.log(self.N(self.current[0])) / self.N(self.current))

                Qli = []
                for a in actions:

                    Q = q +
                    Qli.append(Q)

                maxQ = max(Qli)
                actionCandi = np.argwhere(Node.qTable[self.current[0], :] == maxQ).flatten()
                action = random.choice(np.intersect1d(nextActions, actionCandi))
                bestAction =

                action = np.argmax(legalActions)


            # action = self.chooseAction(heuristic=True, state=state)
            # # update current state for the parent
            # self.current = (action, 0)  # root
            # print("chosen action: ", action)
            # newNode = Node(prbIdx=None, current=self.current, parent=self)
            # self.expand(newNode)
            #
            # while self.isFullyExpanded(state=state, traverse=self.traverse) == (True, False):
            #     action = self.chooseAction(heuristic=True, state=state)
            #     # Expand
            #     self.current = (self.current[0], action)
            #     print("chosen action: ", action)
            #     newNode = Node(prbIdx=None, current=self.current, parent=self)
            #     self.expand(newNode)

    def expand(self, newNode):
        """
        Make a new child to append to a self.children list with either a state or state, action pairs.
        Do I need to append children or just a current state?
        """
        Node.visits[self.current] += 1
        # root visits
        Node.visits[(-1, -1)] += 1
        self.children.append(newNode)
        print("self.children:", self.children)

    def rollout(self, policy=None):
        """
        :input: s', a' d
        :return: R
        """
        if self.depth == 0:
            nextActions = np.argwhere(self.monitor[:, 0] == 0).flatten()  # element
            action = random.choice(nextActions)
            self.current = (action, 0)

        elif self.depth == 1:
            nextActions = np.argwhere(self.monitor[self.current[0], 0] == 0).flatten()  # element
            action = random.choice(nextActions)
            self.current = (self.current[0], action)
        else:
            nextActions = self.cardAvail[self.current[0] * 4 + self.current[1]]
            action = random.choice(nextActions)
            self.current = self.leafState[action]  # end-reward

        numCards = self.contextM[self.current]
        if numCards == 0:
            r = 0
        else:
            r = 1/numCards
        R = r * self.gamma ^ self.depth
        return R

    def backprop(self, reward):
        #if self.isFullyExpanded(state=self.monitor, traverse=self.traverse) == (True, True):
        # print("len(self.children)", len(self.children))
        for child_i, child in enumerate(self.children):  # Better to divide the depth? But the depth is already coded in the coordinates.
            # print("this is the child", child.current)
            Node.visits[child.current] += 1
            self.updateQtable(reward=reward)

            if child.parent is not None:
                child.parent.visits[child.parent] += 1


