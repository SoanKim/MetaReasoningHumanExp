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

        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 0          before selecting a leaf val: depth 2
        | 0 | 0 | 1 | 2 | 3 |   sum(1st col) == 3: depth 1         after selecting a leaf val: depth 3
        | 1 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) < 12: depth 1    after receiving reward = terminal
        | 2 | 0 | 1 | 2 | 3 |   sum(2nd-4th cols) == 12: depth 2
        |---|---|---|---|---|
        """

    def isFullyExpanded(self, monitor):  # checking if it reached the leaf state
        if monitor is None:
            monitor = self.monitor
        else:
            monitor = monitor
        rowSum = [sum(row) for row in monitor]
        if np.sum(monitor[:, 0]) < 3 and 5 not in rowSum:
            self.depth = 0  # num of children = 3
            return False, False
        elif np.sum(monitor[:, 0]) == 3 and 5 not in rowSum:
            self.depth = 1  # num of children = 4
            return True, False
        elif np.sum(monitor[:, 0]) == 3 and 5 in rowSum:  # any(li) means all elements of list are True
            self.depth = 2  # num of cards varies across trials
            return True, True

    def horiLegalMoves(self, monitor):
        if self.isFullyExpanded(monitor=monitor) == (False, False):
            self.depth = 0
            actions = np.argwhere(monitor[:, 0] == 0).flatten()
            transProb = np.take(Node.pTable[:, 0], actions).flatten()
        elif self.isFullyExpanded(monitor=monitor) == (True, False):
            self.depth = 1
            actions = np.argwhere(monitor[self.current[0], :] == 0).flatten()
            transProb = np.take(Node.pTable[self.current[0], :], actions).flatten()
        else:
            self.depth = 2
            actions = self.cardAvail[self.current[0] * 4 + self.current[1]]  # card list
            transProb = np.arange(actions)  # from left to right
        print("self.depth:", self.depth)
        print("actions:", actions)
        print("transProb:", transProb)
        return actions, transProb

    def expand(self):  # add tree policy later
        """
        expanding one step further with a real monitor
        """
        if self.depth == 0:
            nextActions = np.argwhere(self.monitor[self.current[0], :] == 0).flatten()
        elif self.depth == 1:
            nextActions = self.selectCard()
        else:
            nextActions = None

        if nextActions:
            action = self.select(legalActions=nextActions)
        else:
            return None
        return action

    def traverse(self, parent, rolloutPolicy: bool):  # It needs to be done with tempMonitor
        """
        todo: visit all node at least once at the same depth and roll out
        :return: Node.visit[node.current] += 1, reward
        """

        while np.sum(self.monitor[:, 0]) < 4:
            print("np.sum(self.monitor[:, 0])", np.sum(self.monitor[:, 0]))
            action = self.select(monitor=self.monitor)
            self.monitor[self.current] = 1
            print("action", action)
            print("position", self.current)
            print("this will be forever 1")

        # while np.sum(self.monitor[:, 0]) == 3 and np.sum(self.monitor[self.current[0], :]) < 5:
        #     action = self.select(monitor=self.monitor)
        #     self.monitor[self.current] = 1
        #     print("this will be forever 2")

        print("self.monitor", self.monitor)


        # while self.isFullyExpanded(monitor=tempMonitor) == (True, False):
        #     action = self.select(legalActions=legalActions, probs=probs)
        #     tempMonitor[self.current] = 1


            # rwd = self.rollout(final=False, legalActions=legalActions)
            # print("Rwd", rwd)



        # bestAction = self.select(legalActions=legalActions, probs=None)
        # child = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
        # child.Q += rwd
        # self.children.append(child)

    def select(self, monitor):
        """
        todo: choose one action among current depth
        input: current state
        :return: next state and depth + 1
        """

        if np.sum(monitor[:, 0]) < 4:
            self.depth = 0
            legalMoves = np.argwhere(self.monitor[:, 0] == 0).flatten()
            transProb = np.take(Node.pTable[:, 0], legalMoves).flatten()
            ucbRange = Node.ucbTable[:, 0]
        elif np.sum(monitor[:, 0]) == 3 and np.sum(monitor[self.current[0], :] < 5):
            self.depth = 1
            legalMoves = np.argwhere(self.monitor[self.current[0], :] == 0).flatten()
            transProb = np.take(Node.pTable[self.current[0], :], legalMoves).flatten()
            ucbRange = Node.ucbTable[self.current[0], :]
        else:
            self.depth = 2
            legalMoves = np.arange(12)
            transProb = np.zeros((12,))
            ucbRange = None
        print("self.depth", self.depth)
        print("self.current", self.current)

        legalUCBmat = np.take(ucbRange, legalMoves)
        print("legalUCBmat", legalUCBmat)
        maxUCB = max(legalUCBmat)
        actions = np.argwhere(legalUCBmat == maxUCB).flatten()
        print("actions", actions)

        if len(actions) > 1:
            action = [legalMoves[i] for i in np.argsort(transProb)][0]
        else:
            action = np.random.choice(actions)
        print("action", action)

        # update the location
        if self.depth == 0:
            self.current = (action, 0)
        elif self.depth == 1:
            self.current = (self.current[0], action)
        else:
            self.current = None
        print("self.current", self.current)

        # update the current visit counts

        Node.visits[self.current] += 1
        self.monitor[self.current] = 1

        return action

    def rollout(self, final: bool):
        """
        todo: choose actions recursively to get the reward from the leaf node
        :input: s', a' d
        :return: R from depth 2
        """
        # Find the siblings unvisited at the current depth.
        # Check depth by monitor
        # if np.sum(self.monitor[:, 0]) < 1:  # depth 0
        #     element = self.select(legalActions=legalActions, probs=None)
        #     self.current = (element, 0)
        #     print("element: ", element)
        #     self.monitor[element, 0] = 1
        #
        # elif np.sum(self.monitor[:, 0]) == 1 and np.sum(self.monitor[self.current[0], :]) < 1:
        #     action = self.select(legalActions=legalActions, probs=None)
        #     self.monitor[self.current[0], action] = 1
        #     print("action: ", action)

        numCards = self.contextM[self.current]
        if numCards == 0:
            r = 0
        else:
            r = 1 / numCards
        if final:
            cardChosen = self.selectCard()
            rwd = self.getReward(cardChosen)
            R = r + r * pow(self.gamma, self.depth) * rwd
        else:
            R = r + r * pow(self.gamma, self.depth)
        return R

    def selectCard(self):
        assert self.depth == 2, "You cannot select card less than depth 2."
        finalAction = self.current[0] * 4 + self.current[1]
        cardChosen = self.cardAvail[finalAction]
        return cardChosen

    def getReward(self, cardChosen):
        return self.leafState[cardChosen]

    def backprop(self, reward):
        self.monitor[self.current] = 1
        Node.visits[self.current] += 1

        for child_i, child in enumerate(self.children):
            Node.visits[child.current] += 1
            self.updateUCB(reward=reward)
            self.N += 1
            self.Q += reward

            if child.parent is not None:
                Node.visits[child.parent.current] += 1

    def updateUCB(self, reward):  #parent's current?
        childN = Node.visits[self.current]
        Node.visits[self.current] += 1
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
        if childN == 0:
            ucb = np.inf
        else:
            ucb = Node.qTable[self.current] + self.exploreConstant * math.sqrt(math.log(parentN) / childN)
        print("ucb:{}".format(ucb))
        Node.ucbTable[self.current] = Node.ucbTable[self.current] + ucb
        # update the root as well
        Node.ucbTable[self.current[0], 0] = Node.ucbTable[self.current[0], 0] + ucb
        # print("self.contextM", self.contextM)
        print("Node.ucbTable", Node.ucbTable)
        print("Node.qTable", Node.qTable)
