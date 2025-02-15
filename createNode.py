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
        self.ucbTable = np.full((3, 5), np.inf)

        # initialize Q table
        self.qTable = np.zeros((3, 5))

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

    def isFullyExpanded(self, state):
        """
        Fully expanded = leaf state
        """
        if self.depth == 0:
            print("depth is zero and not yet fully expanded")
            return np.sum(state[:, 0]) == 3
        else:
            print("depth is non zero and not yet fully expanded")
            return any(sum(row) == 5 for row in state)


    def bestUCB(self):
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

    def updateQtable(self, node, value, alpha=0.1):
        delta = (1 / (Node.visits[node.current])) * (value - self.qTable[node.current])
        self.qTable[node.current] = self.qTable[node.current] + alpha * delta

    def select(self, policy: bool):  # select a child among children
        # navi is a state for simulation
        tempState = deepcopy(self.navi)

        # make a child node at every position
        # if root, both current status and action available are different.
        bestChild = None
        bestUCB = -np.inf

        if self.depth == 0:  # element of the root
            print("================ depth is zero =================")
            parent = Node()
            while not self.isFullyExpanded(tempState):  # by depth
                elementAvail = np.argwhere(tempState[:, 0] == 0).flatten()
                if not policy:
                    element = random.choice(elementAvail)
                else:
                    element = None
                    bestQval = 0
                    for elem in elementAvail:
                        qVal = self.qTable[elem, 0]
                        if qVal > bestQval:
                            bestQval = qVal
                            element = elem
                    if element is None:
                        element = random.choice(elementAvail)
                print("element: {}".format(element))

                # init a child node
                child = Node(current=(element, 0), parent=parent)

                # update the state on the navi as well as tempState
                self.navi[element, 0] = 1
                tempState[element, 0] = 1
                value = self.rollout(child)

                # backup to make sure the visitCount is not zero
                self.backprop(value=value, node=child)

                # add child
                parent.children.append(child)

                # update Qtable: update the value and the visits --> refer to Dr. Miller
                self.updateQtable(node=child, value=value)

                # update UCB table
                ucb = child.Q + self.exploreConstant * math.sqrt(math.log(child.parent.N) / child.N)
                self.ucbTable[element, 0] = ucb

                if ucb > bestUCB:
                    bestChild = child
                    bestUCB = ucb
                    bestChild.depth += 1
            print("qTable", self.qTable)
            print("")
            print("ucbTable", self.ucbTable)
            print("")
            print("value", self.contextM)
            print("")
            print("bestChild.Q", bestChild.Q)
            print("")
            print("navi", self.navi)
            print("")
            print("tempState", tempState)
            print("")
            print("bestChild.current", bestChild.current)

        else:
            print("++++++++++++++++++++ depth is Non zero +++++++++++++++++")
            parent = self
            print("parent.Q", self.parent.Q)
            while not self.isFullyExpanded(tempState):  # by depth
                element = self.current[0]
                actionAvail = np.argwhere(tempState[element, :] == 0).flatten()

                if not policy:
                    action = random.choice(actionAvail)
                else:
                    action = None
                    bestQval = 0
                    for act in actionAvail:
                        qVal = self.qTable[element, act]
                        if qVal > bestQval:
                            bestQval = qVal
                            action = act
                    if action is None:
                        action = random.choice(actionAvail)
                print("action: {}".format(action))

                child = Node(current=(element, action))
                child.current = (element, action)

                # update the state on the tempState
                tempState[self.current[0], action] = 1
                value = self.rollout(child)

                # backup
                self.backprop(value=value, node=child)

                # add child
                parent.children.append(child)

                # update Qtable: update the value and the visits --> refer to Dr. Miller
                self.updateQtable(node=child, value=value)

                # ucb Table
                ucb = child.Q + self.exploreConstant * math.sqrt(math.log(child.parent.N) / child.N)
                self.ucbTable[element, action] = ucb

                if ucb > bestUCB:
                    bestChild = child
                    bestUCB = ucb
                    bestChild.depth += 1

            print("qTable", self.qTable)
            print("")
            print("ucbTable", self.ucbTable)
            print("")
            print("value", self.contextM)
            print("")
            print("bestChild.Q", bestChild.Q)
            print("")
            print("navi", self.navi)
            print("")
            print("tempState", tempState)
            print("")
            print("bestChild.current", bestChild.current)
        return bestChild

    def expand(self, bestChild):
        """
        expand - rollout - backprop
        """
        # store the bestChild's coordinate on the invariable state
        element = bestChild.current[0]
        actionAvail = np.argwhere(self.navi[element, :] == 0).flatten()


        action = random.choice(actionAvail)
        newNode = Node(current=(element, action), parent=bestChild)

        return newNode

    def rollout(self, node):
        leafValue = 1 / self.contextM[node.current] if self.contextM[node.current] != 0 else 0
        return leafValue

    def backprop(self, value, node):
        # update visits of the Q table
        Node.visits[node.current] = Node.visits[node.current] + 1

        # update the parent
        node.parent.Q += value
        node.parent.N += 1

        # update the child
        node.N += 1
        node.Q += value


# if __name__ == '__main__':
#
#     game = Game()
#     contextM, cardAvail, answer, navi, leafState = game.prbInit()
#     root = Node()
#     print(root.parent)
