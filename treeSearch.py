#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 20:05 on 25/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
import numpy as np
from createNode import Node
from collections import defaultdict
import math
import random


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

        # state
        self.depth = 0
        self.monitor = np.zeros((3, 5))

        # hyperparam
        self.gamma = 1
        self.exploreConstant = 2

        # root node should be executed inside of MCTS to get updated.
        self.root = Node(current=None, parent=None)

        # nested_dict['person1']['name'] = 'Alice'

    def isFullyExpanded(self, node):
        if np.all(self.monitor[:, 0]) == 0:
            return False, False
        elif np.all(self.monitor[:, 0]) == 1 and np.all(self.monitor[node.current[0], 1:]) == 0:
            return True, False
        elif np.all(self.monitor[:, 0]) == 1 and np.any(self.monitor[node.current[0], 1:]) == 1:
            return True, True

    def getState(self, node):
        # assert self.depth < 3, "It is depth 3."
        if self.depth == 0:
            state = np.argwhere(self.monitor[:, 0] == 0).flatten()
            prob = MCTS.ucbTable[state, 0]
        else:
            state = np.argwhere(self.monitor[node.current[0], 1:] == 0).flatten() + 1
            prob = MCTS.ucbTable[node.current[0], state]
        return state, prob

    def getLeafVal(self, current):
        assert current is not None, "This is the root state"
        rwd = self.contextM[current]
        if rwd != 0:
            reward = 1 / rwd
        else:
            reward = 0
        return reward

    def isLeaf(self, node):
        return len(node.children) == 0

    def argMaxAction(self, state, probs):  # greedy action
        if len(set(probs)) == 1:
            bestAction = random.choice(state)
        else:
            bestAction = np.argmax(probs)
        return bestAction

    def selectArm(self, node):
        state, prob = self.getState(node)
        print(f"state {state}")
        print(f"prob {prob}")
        assert len(state) == len(prob), "The length of state and action doesn't match"
        action = self.argMaxAction(state, prob)
        return action

    def updateQ(self, current, reward):
        # Guez et al., 2012
        delta = reward - MCTS.qTable[current] / MCTS.visits[current]
        MCTS.qTable[current] += delta

    def updateUCB(self, current, node):
        # print(f"node.parent.N: {node.parent.N}")
        # print("child.dict", node.__dict__)
        ucb = MCTS.qTable[current] + self.exploreConstant * math.sqrt(math.log(node.parent.N) / MCTS.visits[current])
        node.ucb = ucb
        # overwrite ucb
        MCTS.ucbTable[current] = ucb

    def backprop(self, reward, node):
        # update the parent node
        node.parent.N += 1
        node.parent.R += reward

        if node.parent.current is not None:
            MCTS.visits[node.parent.current] += 1
            MCTS.freqTable[node.parent.current] += 1
            self.monitor[node.parent.current] += 1
            self.updateQ(current=node.parent.current, reward=reward)
            self.updateUCB(current=node.parent.current, node=node)

        # update the current node
        node.N += 1
        node.R += reward
        MCTS.visits[node.current] += 1
        MCTS.freqTable[node.current] += 1
        self.monitor[node.current] += 1
        self.updateQ(current=node.current, reward=reward)
        self.updateUCB(current=node.current, node=node)

        # append child to parent node
        node.parent.children.append(node)
        print("node.parent.children:", node.parent.children)

        print("monitor\n", self.monitor)
        print("visits\n", MCTS.visits)
        print("qTable\n", MCTS.qTable)
        print("freqTable\n", MCTS.freqTable)
        print("ucbTable\n", MCTS.ucbTable)

    def rollout(self, node):
        bestAction = self.selectArm(node)
        print(f"bestAction: {bestAction}")
        current = self.markCurrent(bestAction, node=node)
        print(f"current: {current}")
        child = Node(current=current, parent=node)
        leafVal = self.getLeafVal(current=current)
        self.backprop(reward=leafVal, node=child)
        print(f"leafValue: {leafVal}")
        return leafVal, current

    def traverse(self, node):
        """
        todo: select actions to the leaf node (depth 2) by greedy policy
        rollout once --> visit += 1
        :return: nothing. But update reward of each child.
        """
        if node:
            node = node
        else:
            node = self.root

        if self.depth == 0:
            while len(self.getState(node)[0]) > 0:
                print(f" ============== Starting depth {self.depth} =============")
                reward, current = self.rollout(node)
            self.depth += 1
        else:
            print(" ##################### depth 1 starts ####################")
            if np.all(self.monitor[node.current[0], 1:]) == 0:
                reward, current = self.rollout(node)
            self.depth += 1
            while self.getBestRoot() is not None:
                newParent = self.getBestRoot()
                print(f" $$$$$$$$$$$$$ newParent $$$$$$$$$$$$", newParent.current)
                reward, current = self.rollout(newParent)
            self.depth += 1
            finalReward = self.selectCard(current)

    def getBestChild(self, node):
        assert self.depth < 3, "depth must be less than 3 to choose the best child."
        if node:
            node = node
        else:
            node = self.root

        if self.depth <= 1:
            maxUCB = np.argmax(MCTS.ucbTable[:, 0])
            maxUCBposition = (maxUCB, 0)
        else:
            maxUCB = np.argmax(MCTS.ucbTable[node.current[0], 1:]) + 1
            maxUCBposition = (node.current[0], maxUCB)
        print(f"maxUCB: {maxUCB}")
        print(f"maxUCBposition: {maxUCBposition}")

        for child in node.children:
            if child.current == maxUCBposition:
                node.current = child.current
                return child

    def getBestRoot(self):
        bestRoot = np.argmax(self.ucbTable[:, 0])
        for child in self.root.children:
            if child.current == (bestRoot, 0):
                if self.isLeaf(child):
                    print(f"This node {child.current} has never been expanded!")
                    return child

    def getBestParent(self):
        bestRootLi = []
        bestRoot1 = self.getBestRoot()
        if bestRoot1 is not None:
            bestRootLi.append(bestRoot1.current[0])
            self.traverse(node=bestRoot1)
            bestRoot2 = self.getBestRoot()
            if bestRoot2 is not None:
                bestRootLi.append(bestRoot2.current[0])
                self.traverse(node=bestRoot2)
        print("bestRootLi", bestRootLi)
        return bestRootLi

    def markCurrent(self, action, node):
        if type(action) == np.ndarray:
            action = action.item()
        elif type(action) == list:
            action = action[0]
        else:
            action = action

        if self.depth == 0:
            current = (action, 0)
        else:
            current = (node.current[0], action)
        return current

    def isTerminal(self):
        return np.sum(self.monitor) == 15

    def getReward(self, cardChosen):
        reward = self.answer == cardChosen
        return int(reward)

    def selectCard(self, bestChildCurrent):
        #assert self.depth == 3, f"You cannot select card before depth 3. current depth: {self.depth}"
        print("cards avail", self.cardAvail)
        cardsAvail = self.cardAvail[bestChildCurrent[0]][bestChildCurrent[1] - 1]
        if len(cardsAvail) == 0:
            print("There is no card available!")
            bestParent = np.argmax(self.monitor[:, 0])
            for child in self.root.children:
                if child.current == (bestParent, 0):
                    return self.traverse(child)

            # actions = np.argwhere(self.monitor[bestParent, 1:] == 0).flatten() + 1
            # action = random.choice(actions)
            # cardsAvail = self.cardAvail[bestParent][action - 1]
            # cardChosen = random.choice(cardsAvail)
        else:
            cardChosen = random.choice(cardsAvail)
            print("cardChosen: {cardChosen}")
            reward = self.getReward(cardChosen)
            print(f"***********  reward: {int(reward)}  ***********")
            return reward
