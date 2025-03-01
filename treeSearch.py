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

    def getDepth(self):  # update depth
        if np.all(self.monitor[:, 0]) == 0:
            self.depth = 0
        elif np.all(self.monitor[:, 0]) == 1:
            self.depth = 1
        elif np.all(self.monitor[:, 0]) == 1 and np.any(self.monitor[self.node.current[0], 1:]) == 1:
            self.depth = 3
        else:
            self.depth = 2
        return self.depth

    def getReward(self, current):
        rwd = self.contextM[current]
        if rwd != 0:
            reward = 1 / rwd
        else:
            reward = 0
        return reward

    # def isFullyExpanded(self, rollout: bool):  # checking if it reached the leaf
    #     if rollout is True:
    #         idx1 = 1
    #         idx2 = 2
    #     else:
    #         idx1 = 3
    #         idx2 = 5
    #
    #     rowSum = [sum(row) for row in self.monitor]
    #     if np.sum(self.monitor[:, 0]) < idx1 and idx2 not in rowSum:
    #         self.depth = 1  # num of children = 3
    #         return False, False
    #     elif np.sum(self.monitor[:, 0]) == idx1 and idx2 not in rowSum:
    #         self.depth = 2  # num of children = 4
    #         return True, False
    #     elif np.sum(self.monitor[:, 0]) == idx1 and idx2 in rowSum:  # any(li) means all elements of list are True
    #         self.depth = 3  # num of cards varies across trials
    #         return True, True

    def argMaxAction(self, actions, probs):  # greedy action
        if len(actions) > 0:
            best_action_index = np.argmax(probs)
            action = actions[best_action_index]
            return action

    def select(self):
        depth = self.getDepth()
        if depth == 0:
            probs = MCTS.ucbTable[:, 0]
            actions = np.arange(len(probs))
            action = self.argMaxAction(actions, probs)
            self.node.current = (action, 0)
            self.monitor[self.node.current] = 1
            parent = self.node
            newChild = Node(current=self.node.current, parent=parent)
            print("node.current of best child", self.node.current)
        elif depth == 1:
            probs = MCTS.ucbTable[self.node.current[0], :]
            actions = np.arange(len(probs))
            action = self.argMaxAction(actions, probs)
            self.node.current = (self.node.current[0], action)
            self.monitor[self.node.current] = 1
            probs = MCTS.ucbTable[self.node.current[0], 1:]
            bestAction = self.argMaxAction(actions=actions, probs=probs)
            bestCurrent = (self.node.current[0], bestAction)
            self.node.current = bestCurrent
            parent = self.expand()
            newChild = Node(current=self.node.current, parent=parent)
        else:
            return None
        return newChild

    def updateQ(self, current, reward):
        # Guez et al., 2012
        delta = reward - MCTS.qTable[current] / MCTS.visits[current]
        MCTS.qTable[current] = MCTS.qTable[current] + delta
        print(f"Q: {MCTS.qTable[current] + delta}")
        print("MCTS.qTable \n", MCTS.qTable)

    def updateUCB(self, current):
        ucb = MCTS.qTable[current] + self.exploreConstant * math.sqrt(math.log(self.node.N) / MCTS.visits[current])
        # overwrite ucb
        MCTS.ucbTable[current] = ucb
        print(f"UCB: {ucb}")
        print("MCTS.ucbTable \n", MCTS.ucbTable)

    def backprop(self, child, reward, rollout: bool):
        # update parent(root)'s N whenever visiting.
        print("%%%% PARENT", child.parent)
        if child.parent is not None:
            child.parent.N += 1
            child.parent.Q += reward
            MCTS.qTable[child.parent.current] += reward
            MCTS.visits[child.parent.current] += 1
            MCTS.freqTable[child.parent.current] += 1
            self.updateQ(current=child.parent.current, reward=reward)
            self.updateUCB(current=child.parent.current)
            print(f"parent's N: {child.parent.N}")
            print(f"parent's Q: {child.parent.Q}")

        child.Q += reward
        child.N += 1

        if rollout is False:
            self.updateQ(current=child.current, reward=reward)
            self.monitor[child.current] = 1
            MCTS.freqTable[child.current] += 1
        print("freqTable\n", MCTS.freqTable)
        print("visits\n", MCTS.visits)
        print("qTable\n", MCTS.qTable)
        print("ucbTable\n", MCTS.ucbTable)
        print("monitor\n", self.monitor)

    def rollout(self):
        """
        todo: select actions to the leaf node (depth 2) by greedy policy
        rollout once --> visit += 1
        :return: rwd
        """
        depth = self.getDepth()
        if depth == 0:
            while not np.all(self.monitor[:, 0]) == 1:
                print(" =============== FIRST LEVEL =============")
                child = self.select()
                self.node.children.append(child)
                # get rewards
                reward = self.getReward(current=child.current)
                self.backprop(child=child, reward=reward, rollout=True)
        elif depth == 1:
            print("######### self.node.current[0] #########", self.node.current[0])
            while not np.all(self.monitor[self.node.current[0], 1:]) == 1:
                print(" $$$$$$$$$$ NEXT LEVEL $$$$$$$$$$$$")
                child = self.select()  # HOw do you load best child?
                self.node.children.append(child)
                # get rewards
                reward = self.getReward(current=child.current)
                self.backprop(child=child, reward=reward, rollout=True)

    def expand(self):
        depth = self.getDepth()
        if depth == 1:
            maxUCB = np.argmax(MCTS.ucbTable[:, 0])
            maxUCBposition = (maxUCB, 0)
            for child in self.node.children:
                if child.current == maxUCBposition:
                    self.node.current = child.current
                return child

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
