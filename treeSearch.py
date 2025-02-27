#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 20:05 on 25/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
import numpy as np
from createNode import Node
from collections import defaultdict
from copy import deepcopy
import math


class MCTS:
    visits = defaultdict(lambda: 0)  # please initialize this every trial
    qTable = np.zeros((3, 5))
    ucbTable = np.full((3, 5), np.inf)
    freqTable = np.zeros((3, 5))  # prob of successful visits

    def __init__(self, prbIdx):
        # initialize game
        super().__init__()
        self.prbIdx = 0 if prbIdx is None else prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.leafState = self.game.prbInit()

        # initialize self.node
        self.node = Node(current=None, parent=None)

        # state
        self.depth = 0
        self.monitor = np.zeros((3, 5))

        # hyperparam
        self.gamma = 1
        self.exploreConstant = 2

    def addChild(self, current, parent):
        child = Node(current, parent)
        child.N = 1
        self.node.children.append(child)

    def getReward(self, current):
        rwd = self.contextM[current]
        if rwd != 0:
            reward = 1 / rwd
        else:
            reward = 0
        return reward

    def isFullyExpanded(self, rollout: bool):  # checking if it reached the leaf
        if rollout is True:
            idx1 = 1
            idx2 = 2
        else:
            idx1 = 3
            idx2 = 5

        rowSum = [sum(row) for row in self.monitor]
        if np.sum(self.monitor[:, 0]) < idx1 and idx2 not in rowSum:
            self.depth = 1  # num of children = 3
            return False, False
        elif np.sum(self.monitor[:, 0]) == idx1 and idx2 not in rowSum:
            self.depth = 2  # num of children = 4
            return True, False
        elif np.sum(self.monitor[:, 0]) == idx1 and idx2 in rowSum:  # any(li) means all elements of list are True
            self.depth = 3  # num of cards varies across trials
            return True, True

    def select(self, actions, probs):  # greedy action
        if len(actions) > 0:
            best_action_index = np.argmax(probs)
            action = actions[best_action_index]
            return action

    def bestChild(self):
        depth = self.getDepth()
        print(f"depth: {depth}")
        if depth <= 1:
            probs = MCTS.ucbTable[:, 0]
            actions = np.arange(len(probs))
            action = self.select(actions, probs)
            self.node.current = [action, 0]
        elif depth == 2:
            probs = MCTS.ucbTable[self.node.current[0], :]
            actions = np.arange(len(probs))
            action = self.select(actions, probs)
            self.node.current = [self.node.current[0], action]
        else:
            return None
        print(f"current: {self.node.current}")
        print(f"bestChild: {action}")
        # print(len(self.node.children))
        # for child_i, child in enumerate(self.node.children):
        #     print("<<<<<<<<<<<< new child >>>>>>>>>>>>>>>")
        #     print(f"child.current: {child.current}")
        #     print(f"child.Q: {child.Q}")
        #     print(f"child.N: {child.N}")

    def rollout(self):
        tempMonitor = deepcopy(self.monitor)
        while not np.all(tempMonitor[:, 0] == 1):
            elements = np.argwhere(tempMonitor[:, 0] == 0).flatten()

            # select by greedy policy on Qs
            probs = np.take(MCTS.qTable[0, :], elements).flatten()
            element = self.select(elements, probs)

            # update temporarily on current, monitor, visit, and Q
            self.node.current = (element, 0)
            tempMonitor[element, 0] = 1
            self.monitor[element, 0] = 1

            # update root.N
            self.node.N += 1
            print(" ############ self.node.N #############", self.node.N)
            # make child nodes
            self.addChild(current=self.node.current, parent=self.node)

            # record permanently
            MCTS.visits[self.node.current] += 1
            MCTS.freqTable[self.node.current] += 1
            actions = np.argwhere(tempMonitor[element, 1:] == 0).flatten() + 1  # offset actions by 1 to get correct idx

            # prevent errors for empty actions
            if len(actions) > 0:
                # select by greedy policy on Qs
                probs = np.take(MCTS.qTable[0, actions], elements).flatten()
                action = self.select(actions, probs)

                # update just temporary because it'll be deleted.
                tempMonitor[element, action] = 1
                # print(f"Selected action (2nd-5th columns): {action}")
            else:
                print("No actions available in columns 1-4 for row:", element)
            print("MCTS.freqTable:\n", MCTS.freqTable)

        coordinates = list(map(tuple, np.argwhere(tempMonitor[:, 1:] != 0)))

        for coordi in coordinates:
            rwd = self.getReward(current=coordi)
            self.updateQ(coordi=coordi, reward=rwd)
            self.updateUCB(coordi=coordi)

            for child in self.node.children:
                if child.current == coordi:
                    child.Q = rwd

        print("MCTS.qTable:\n", MCTS.qTable)
        print("MCTS.ucbTable:\n", MCTS.ucbTable)

    def updateQ(self, coordi, reward):
        # Guez et al., 2012
        delta = reward - MCTS.qTable[coordi] / MCTS.visits[coordi]
        MCTS.qTable[coordi] = MCTS.qTable[coordi] + delta

    def updateUCB(self, coordi):
        ucb = MCTS.qTable[coordi] + self.exploreConstant * math.sqrt(math.log(self.node.N) / MCTS.visits[coordi])
        # overwrite ucb
        MCTS.ucbTable[coordi] = ucb

    def getDepth(self):  # update depth
        if all(self.monitor[:, 0]) == 0:
            self.depth = 0
        elif 0 < np.sum(self.monitor[:, 0]) < 4:
            self.depth = 1
        elif 0 < np.sum(self.monitor[:, 0]) < 4 and np.all(np.sum(self.monitor[:, 1:]), axis=1) == 1:
            self.depth = 3
        else:
            self.depth = 2
        return self.depth

    def markCurrent(self, action):
        if type(action) == np.ndarray:
            action = action.item()
        elif type(action) == list:
            action = action[0]
        else:
            action = action

        if self.depth <= 1:
            self.node.current = (action, 0)
        elif self.depth == 2:
            self.node.current = (self.node.current[0], action)



    # def backprop(self, reward, traverse: bool):
    #     if traverse == True:
    #
    #         if self.node.current != (np.nan, np.nan):  # default
    #             self.monitor[self.node.current] = 1
    #         self.node.Q += reward
    #         self.node.N += 1
    #         self.node.visits[self.node.current] += 1

    # def traverse(self):
    #     """
    #     todo: visit all self.node at least once at the same depth and roll out
    #     :return: Node.visit[self.node.current] += 1, reward
    #     """
    #
    #     print(MCTS.qTable)

    # choices = set(list(map(tuple, np.argwhere(tempState == 1))))
    # children = set([child.current for child in self.node.children])
    # rolledOut = choices.difference(children)
    #
    # for row, col in rolledOut:
    #     rwd = self.getReward(current=(row, col))
    #     parent = (row, col-1)

    # MCTS.visits[choice]
    #
    # self.qTable[choice] = rwd

    #     print("element", element)
    #
    #
    #     print("self.node.current", self.node.current)
    # else:
    #
    #     action = self.select(a, p)
    #     self.node.current = (self.node.current[0], action)
    #     self.monitor[self.node.current] = 1

    # def isFullyExpanded(self, traverse):
    #     """
    #     This is the case of expanding. Not traversing. In this case, you count children.
    #     """
    #     if traverse is False:
    #         if any(np.sum(self.monitor, axis=1)) == 2:
    #             return True
    #         else:
    #             return False
    #     else:
    #         if all(np.sum(self.monitor, axis=1)) == 2:
    #             return True
    #         else:
    #             return False

    # def expand(self, traverse=False):
    #     """
    #     todo: switch state by depth
    #     """
    #     if self.depth == 0:
    #         action = self.select(self.node, traverse=traverse)
    #         self.markCurrent(self.node=self.node, element=action, action=0)
    #         print("self.node.current 333333")
    #         print(self.node.current)
    #         assert action != None
    #         self.addChild(self.node=self.node, action=action)
    #         rwd = self.rollout()
    #         print("rwd", rwd)
    #         self.backprop(reward=rwd, self.node=)

    # def isTerminal(self):
    #     print("sum", np.sum(self.monitor))
    #     return np.sum(self.monitor) == 15



    #
    # def getMeanValue(self):
    #     if self.depth <= 1:
    #         # upper self.node
    #         childNum = len(np.argwhere(self.monitor[self.node.current[0], :] == 1).flatten())
    #         avgChildQ = np.sum(MCTS.qTable[self.node.current[0], :]) / childNum
    #         print("avgChildQ", avgChildQ)
    #         print("childNum", childNum)
    #     else:
    #         avgChildQ = 0
    #     if self.node.parent is None:
    #         MCTS.ucbTable[self.node.current[0], 0] = avgChildQ + 2 * math.sqrt(math.log(self.node.N) / self.node.N)
    #     else:
    #         MCTS.ucbTable[self.node.current[0], 0] = avgChildQ + 2 * math.sqrt(math.log(self.node.parent.N) / self.node.N)
    #
    # def selectCard(self):
    #     assert self.depth == 2, "You cannot select card less than depth 2."
    #     print("cardsavail", self.cardAvail)
    #     cardsAvail = self.cardAvail[self.node.current[0]][self.node.current[1] - 1]
    #     print("cardsAvail", cardsAvail)
    #     cardChosen = random.choice(cardsAvail)
    #     print("cardChosen", cardChosen)
    #     return cardChosen
    #
    # def getReward(self, cardChosen):
    #     return self.leafState[cardChosen]
