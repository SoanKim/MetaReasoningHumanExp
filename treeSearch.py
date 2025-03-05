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
        self.ucbTable = np.full((3, 5), np.inf)

        # nested_dict['person1']['name'] = 'Alice'

    def isFullyExpanded(self, node):
        if np.all(self.monitor[:, 0]) == 0:
            return False, False
        elif np.all(self.monitor[:, 0]) == 1 and np.all(self.monitor[node.current[0], 1:]) == 0:
            return True, False
        elif np.all(self.monitor[:, 0]) == 1 and np.any(self.monitor[node.current[0], 1:]) == 1:
            return True, True

    def getState(self, node):
        #assert self.depth < 3, "It is depth 3."
        if self.depth == 0:
            state = np.argwhere(self.monitor[:, 0] == 0).flatten()
            prob = self.ucbTable[state, 0]
        else:
            state = np.argwhere(self.monitor[node.current[0], 1:] == 0).flatten() + 1
            prob = self.ucbTable[node.current[0], state]
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
        print(f">>>>>> CURRENT, REWARD, VISIT, & DELTA: {current}, {round(reward, 3)}, {MCTS.visits[current]}, {round(delta, 3)} <<<<<<<")
        MCTS.qTable[current] += delta

    def updateUCB(self, current, node):
        # print(f"node.parent.N: {node.parent.N}")
        # print("child.dict", node.__dict__)
        ucb = MCTS.qTable[current] + self.exploreConstant * math.sqrt(math.log(node.parent.N) / MCTS.visits[current])
        node.ucb = ucb
        # overwrite ucb
        self.ucbTable[current] = ucb

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
        #print("node.parent.children:", node.parent.children)
        #print("visits\n", MCTS.visits)
        print("monitor\n", self.monitor)
        print("qTable\n", MCTS.qTable)
        print("freqTable\n", MCTS.freqTable)
        print("ucbTable\n", self.ucbTable)

    def rollout(self, node):  # if you return a value, it stops after one iteration.
        bestAction = self.selectArm(node)
        print(f"bestAction: {bestAction}")
        current = self.getCurrent(bestAction, parent=node)
        print(f"current: {current}")
        child = Node(current=current, parent=node)
        leafVal = self.getLeafVal(current=current)
        self.backprop(reward=leafVal, node=child)
        print(f"leafValue: {leafVal}")
        # return leafVal, current

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
        print(f"depth: {self.depth}")
        # This is the problem !!!!
        if self.depth == 0:
            while len(self.getState(node)[0]) > 0:  # If you change while into if, there's no child
                print(f" ============== Starting depth {self.depth} =============")
                self.rollout(node)  # reward, current
            self.depth += 1
        else:
            print(" ##################### depth 1 starts ####################")
            if np.all(self.monitor[node.current[0], 1:]) == 0:
                self.rollout(node)  # reward, current
                self.depth += 1

            while self.getBestRoot() is not None:
                finalR = 0
                if len(self.getState(node)[0]) == 0:
                    break
                else:
                    print("len(self.getState(node)[0])", len(self.getState(node)[0]))
                    newParent = self.getBestRoot()
                    print(f" $$$$$$$$$$$$$ Going up for the best Parent!! $$$$$$$$$$$$", newParent.current)
                    self.rollout(newParent)
                    bestChild = newParent.children[-1]
                    finalReward = self.leafState[bestChild.current[0]*4+bestChild.current[1]-1]  #self.leafState[bestChild.current[0]*4+bestChild.current[1]-1]
                    print(f"finalReward: {finalReward}")
                    print(f"newParent.current: {newParent.current}")
                    finalR += finalReward
                self.depth += 1
                return finalR

    def getBestChild(self, node):
        assert self.depth < 3, "depth must be less than 3 to choose the best child."
        if node:
            node = node
        else:
            node = self.root

        if self.depth <= 1:
            maxUCB = np.argmax(self.ucbTable[:, 0])
            maxUCBposition = (maxUCB, 0)
        else:
            maxUCB = np.argmax(self.ucbTable[node.current[0], 1:]) + 1
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

    def getCurrent(self, action, parent):
        if type(action) == np.ndarray:
            action = action.item()
        elif type(action) == list:
            action = action[0]
        else:
            action = action

        if self.depth == 0:
            current = (action, 0)
        else:
            current = (parent.current[0], action)
        return current

    def isTerminal(self):
        return np.sum(self.monitor) == 15

    def getReward(self, cardChosen):
        reward = self.answer == cardChosen
        print(f"answer: {self.answer}, cardChosen: {cardChosen}")
        return int(reward)

    def selectCard(self, bestChildCurrent):
        #assert self.depth == 3, f"You cannot select card before depth 3. current depth: {self.depth}"
        print("cards avail", self.cardAvail)
        cardsAvail = self.cardAvail[bestChildCurrent[0]][bestChildCurrent[1] - 1]
        if len(cardsAvail) == 0:
            print("There is no card available!")
            reward = 0
            bestParent = np.argmax(self.monitor[:, 0])
            for child in self.root.children:
                if child.current == (bestParent, 0):
                    self.traverse(child)

            # actions = np.argwhere(self.monitor[bestParent, 1:] == 0).flatten() + 1
            # action = random.choice(actions)
            # cardsAvail = self.cardAvail[bestParent][action - 1]
            # cardChosen = random.choice(cardsAvail)
        else:
            if self.prbIdx < 60:
                cardChosen = random.choice(cardsAvail)
                print(f"cardChosen: {cardChosen}")
                reward = self.getReward(cardChosen)
                print(f"***********  reward: {int(reward)}  ***********")
            else:
                reward = self.getLeafVal(bestChildCurrent)
        return reward
