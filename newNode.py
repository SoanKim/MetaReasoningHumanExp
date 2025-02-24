#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 17:10 on 24/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import random
import numpy as np
from copy import copy, deepcopy
from createGame import *
from collections import defaultdict
import math


# TO DO: USE self.isExpanded  and keep track of the depth instead of manually updating the depth.

class Node:
    visits = defaultdict(lambda: 0)  # please initialize this every trial
    qTable = np.zeros((3, 5))
    ucbTable = np.full((3, 5), np.inf)
    pTable = np.zeros((3, 5))  # prob of successful visits

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
        self.N = 0

        # reward comes from the leaf value
        self.Q = 0

    def getDepth(self):
        rowSum = [sum(row) for row in self.monitor]
        if self.monitor.T[0, :].all() != 1:
            self.depth = 0
        else:
            if 5 not in rowSum:
                self.depth = 1
            else:
                self.depth = 2
        return self.depth

    def getLegalActions(self):
        depth = self.getDepth()
        print("depth", depth)
        if depth == 0:
            actions = np.argwhere(self.monitor[:, 0] == 0).flatten()
            probs = np.take(Node.pTable[0, :], actions).flatten()
        elif depth == 1:
            actions = np.argwhere(self.monitor[self.current[0], :] == 0).flatten()
            probs = np.take(Node.pTable[self.current[0], :], actions).flatten()
        else:
            actions = self.cardAvail[self.current[0]][self.current[1]-1]
            probs = None
        return actions, probs

    def isFullyExpanded(self):
        return self.getDepth == 2

    def getBestChild(self):
        depth = self.getDepth()
        actions, probs = self.getLegalActions()
        if depth == 0:
            ucbRange = self.ucbTable[:, 0]
        elif depth == 1:
            ucbRange = self.ucbTable[self.current[0], :]
        else:
            ucbRange = np.arange(12)
        print("ucbRange", ucbRange)
        bestUCB = np.argwhere(ucbRange == max(ucbRange)).flatten()

        if len(bestUCB) > 1:
            actionLi, probLi = list(zip(*sorted(zip(actions, probs))))
            bestAction = actionLi[0]
        else:
            bestAction = bestUCB
        print("bestAction", bestAction.item())
        return bestAction.item()

    def expand(self):
        child = Node(prbIdx=self.prbIdx, current=self.current, parent=None)
        self.children.append(child)
        self.depth += 1

    # def updateUCB(self):  #parent's current?
    #     childN = Node.visits[self.current]
    #     Node.visits[self.current] += 1
    #     if childN == 0:
    #         Node.ucbTable[self.current] = np.inf
    #     else:
    #         Node.ucbTable[self.current] = reward  # PROBLEMP: UCB table's node N is too high and it's not exploring
    #     if self.depth == 0:
    #         parentN = Node.visits[-1, -1]
    #     elif self.depth == 1:
    #         parentN = Node.visits[self.current[0], 0]
    #     else:
    #         parentN = Node.visits[self.current]
    #
    #     print("parentN: {}, childN: {}".format(parentN, childN))
    #     # delta = reward - Node.qTable[self.current] / childN
    #
    #     Node.qTable[self.current] = Node.qTable[self.current] + reward
    #     # update the root as well
    #     Node.qTable[self.current[0], 0] = Node.qTable[self.current[0], 0] + reward
    #     if childN == 0:
    #         ucb = np.inf
    #     else:
    #         ucb = Node.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parentN) / childN)
    #     print("ucb:{}".format(ucb))
    #     Node.ucbTable[self.current] = Node.ucbTable[self.current] + ucb
    #     # update the root as well
    #     Node.ucbTable[self.current[0], 0] = Node.ucbTable[self.current[0], 0] + ucb
    #     # print("self.contextM", self.contextM)
    #     print("Node.ucbTable", Node.ucbTable)
    #     print("Node.qTable", Node.qTable)

    def traverse(self, parent):
        depth = self.getDepth()
        actions, _ = self.getLegalActions()
        print("self.current", self.current)
        for i in range(len(actions)):
            bestAction = self.getBestChild()
            print("bestAction", bestAction)
            if depth == 0:
                self.current = (bestAction, 0)
                print("self.current 1", self.current)
                nextAction = random.choice(list(range(4)))
                rwd = self.contextM[bestAction, nextAction]
            elif depth == 1:
                self.current = (self.current[0], bestAction)
                print("self.current 2", self.current)
                rwd = self.contextM[self.current[0], bestAction]
            else:
                self.current = None
                rwd = None
            if rwd != 0:
                reward = 1/rwd
            else:
                reward = 0
            self.Q += reward
            self.N += 1
            parent.N += 1
            Node.qTable[self.current] = Node.qTable[self.current] + reward
            print("qTable", Node.qTable)
            if self.N == 0:
                ucb = np.inf
            else:
                ucb = Node.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parent.N) / self.N)
            Node.ucbTable[self.current] = ucb
            print("ucbTable", Node.ucbTable)
            if self.current != (-1, -1):
                self.monitor[self.current] = 1

            child = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
            Node.visits[self.current] += 1
            self.children.append(child)

    # def rollout(self):
    #     #toStart = np.argwhere(~self.monitor[:, 1:].any(axis=1)).flatten()
    #     depth = self.getDepth()
    #     print(depth)


        # np.argwhere(self.monitor)
        # while depth < 2:
        #     if depth == 0:
        #         nextState = np.argwhere(self.monitor[self.current[0], :] == 0).flatten()
        #     elif depth == 1:
        #         nextState = self.cardAvail[self.current[0]][self.current[1]-1]
        #     else:
        #         nextState = None
        #     depth += 1
        #     if nextState is not None:
        #         action = random.choice(nextState)
        #         actions.append(action)
        #         print(action)
        # print(actions)
        # print(self.contextM[actions])









    # def select(self):
    #     best = self.getBestChild()
    #     return best
    #
    # def traverse(self):
    #     while np.sum(self.monitor[:, 0] < 3):
    #         action = self.select()
    #         print("action", action)
