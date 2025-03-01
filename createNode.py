#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *


class Node:

    # all these properties are from temporal values
    def __init__(self, current=None, parent=None):
        # initialize node properties
        self.parent = self if None else parent  # None for root

        # a state node has child nodes (state and action pairs))
        self.children = []  # {}  # defaultdict(lambda: 0)  or [[], []]  or defaultdict(tuple) for self.children[(element, action)] = leafValue

        # track N of the current node
        self.N = 0

        # reward comes from the leaf value
        self.Q = 0

        # row and column coordinates. (NaN, NaN) for root. Keep track of the current location regardless who you are.
        self.current = (np.nan, np.nan) if current is None else current
        """
        |---|---|---|---|---|   sum(1st col) < 3: depth 1          
        | 0 | 1 | 2 | 3 | 4 |   sum(1st col) == 3: depth 2         
        | 1 | 1 | 2 | 3 | 4 |   sum(2nd-5th cols) < 12: depth 2    
        | 2 | 1 | 2 | 3 | 4 |   sum(2nd-5th cols) == 12: depth 3
        |---|---|---|---|---|
        """

    # def horiLegalMoves(self, monitor):
    #     if self.isFullyExpanded(monitor=monitor) == (False, False):
    #         self.depth = 0
    #         actions = np.argwhere(monitor[:, 0] == 0).flatten()
    #         transProb = np.take(Node.pTable[:, 0], actions).flatten()
    #     elif self.isFullyExpanded(monitor=monitor) == (True, False):
    #         self.depth = 1
    #         actions = np.argwhere(monitor[self.current[0], :] == 0).flatten()
    #         transProb = np.take(Node.pTable[self.current[0], :], actions).flatten()
    #     else:
    #         self.depth = 2
    #         actions = self.cardAvail[self.current[0] * 4 + self.current[1]]  # card list
    #         transProb = np.arange(actions)  # from left to right
    #     print("self.depth:", self.depth)
    #     print("actions:", actions)
    #     print("transProb:", transProb)
    #     return actions, transProb
    #

    #

    #
    #
    #     # while self.isFullyExpanded(monitor=tempMonitor) == (True, False):
    #     #     action = self.select(legalActions=legalActions, probs=probs)
    #     #     tempMonitor[self.current] = 1
    #
    #
    #         # rwd = self.rollout(final=False, legalActions=legalActions)
    #         # print("Rwd", rwd)
    #
    #
    #
    #     # bestAction = self.select(legalActions=legalActions, probs=None)
    #     # child = Node(prbIdx=self.prbIdx, current=self.current, parent=parent)
    #     # child.Q += rwd
    #     # self.children.append(child)
    #
    # def select(self, monitor):
    #     """
    #     todo: choose one action among current depth
    #     input: current state
    #     :return: next state and depth + 1
    #     """
    #
    #     if np.sum(monitor[:, 0]) < 4:
    #         self.depth = 0
    #         legalMoves = np.argwhere(self.monitor[:, 0] == 0).flatten()
    #         transProb = np.take(Node.pTable[:, 0], legalMoves).flatten()
    #         ucbRange = Node.ucbTable[:, 0]
    #     elif np.sum(monitor[:, 0]) == 3 and np.sum(monitor[self.current[0], :] < 5):
    #         self.depth = 1
    #         legalMoves = np.argwhere(self.monitor[self.current[0], :] == 0).flatten()
    #         transProb = np.take(Node.pTable[self.current[0], :], legalMoves).flatten()
    #         ucbRange = Node.ucbTable[self.current[0], :]
    #     else:
    #         self.depth = 2
    #         legalMoves = np.arange(12)
    #         transProb = np.zeros((12,))
    #         ucbRange = None
    #     print("self.depth", self.depth)
    #     print("self.current", self.current)
    #
    #     legalUCBmat = np.take(ucbRange, legalMoves)
    #     print("legalUCBmat", legalUCBmat)
    #     maxUCB = max(legalUCBmat)
    #     actions = np.argwhere(legalUCBmat == maxUCB).flatten()
    #     print("actions", actions)
    #
    #     if len(actions) > 1:
    #         action = [legalMoves[i] for i in np.argsort(transProb)][0]
    #     else:
    #         action = np.random.choice(actions)
    #     print("action", action)
    #
    #     # update the location
    #     if self.depth == 0:
    #         self.current = (action, 0)
    #     elif self.depth == 1:
    #         self.current = (self.current[0], action)
    #     else:
    #         self.current = None
    #     print("self.current", self.current)
    #
    #     # update the current visit counts
    #
    #     Node.visits[self.current] += 1
    #     self.monitor[self.current] = 1
    #
    #     return action
    #
    # def rollout(self, final: bool):
    #     """
    #     todo: choose actions recursively to get the reward from the leaf node
    #     :input: s', a' d
    #     :return: R from depth 2
    #     """
    #     # Find the siblings unvisited at the current depth.
    #     # Check depth by monitor
    #     # if np.sum(self.monitor[:, 0]) < 1:  # depth 0
    #     #     element = self.select(legalActions=legalActions, probs=None)
    #     #     self.current = (element, 0)
    #     #     print("element: ", element)
    #     #     self.monitor[element, 0] = 1
    #     #
    #     # elif np.sum(self.monitor[:, 0]) == 1 and np.sum(self.monitor[self.current[0], :]) < 1:
    #     #     action = self.select(legalActions=legalActions, probs=None)
    #     #     self.monitor[self.current[0], action] = 1
    #     #     print("action: ", action)
    #
    #     numCards = self.contextM[self.current]
    #     if numCards == 0:
    #         r = 0
    #     else:
    #         r = 1 / numCards
    #     if final:
    #         cardChosen = self.selectCard()
    #         rwd = self.getReward(cardChosen)
    #         R = r + r * pow(self.gamma, self.depth) * rwd
    #     else:
    #         R = r + r * pow(self.gamma, self.depth)
    #     return R
    #
    # def selectCard(self):
    #     assert self.depth == 2, "You cannot select card less than depth 2."
    #     finalAction = self.current[0] * 4 + self.current[1]
    #     cardChosen = self.cardAvail[finalAction]
    #     return cardChosen
    #
    # def getReward(self, cardChosen):
    #     return self.leafState[cardChosen]
    #
    # def backprop(self, reward):
    #     self.monitor[self.current] = 1
    #     Node.visits[self.current] += 1
    #
    #     for child_i, child in enumerate(self.children):
    #         Node.visits[child.current] += 1
    #         self.updateUCB(reward=reward)
    #         self.N += 1
    #         self.Q += reward
    #
    #         if child.parent is not None:
    #             Node.visits[child.parent.current] += 1
    #
    # def updateUCB(self, reward):  #parent's current?
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
