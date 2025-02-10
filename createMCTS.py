#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from createNode import Node
from copy import deepcopy
import numpy as np
import random
import math


class MCTS:

    def __init__(self, prbIdx):  # game = an indexed problem
        self.explorConstant = 2
        self.prbIdx = prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.navi, self.leafState = self.game.prbInit()
        self.root = Node()
        self.depth = self.root.trackDepth(self.navi)
        self.fullExpand = self.root.isFullyExpanded(self.navi)
        self.legalMoves = self.game.legalMove()

    def traverse(self, rollOutPolicy=None):
        if rollOutPolicy is None:
            tempEnv = deepcopy(self.navi)
            positions = []
            values = []
            while np.sum(tempEnv[:, 0]) < 3:
                # row = actions in depth 1
                rowCandi = np.argwhere(tempEnv[:, 0] == 0).flatten()
                row = random.choice(rowCandi)

                # col = actions in depth 2
                colCandi = np.argwhere(tempEnv[row, :] == 0).flatten()
                colCandi = [c for c in colCandi]
                col = random.choice(colCandi)

                # mark elem choice
                self.navi[row, 0] = 1

                # rollout info is volatile
                tempEnv[row, 0] = 1
                tempEnv[row, col] = 1
                positions.append((row, col))

                # value of those actions
                val = self.contextM[row, col]
                if val != 0:
                    values.append(np.round(1/val, 2))
                else:
                    values.append(0)
            valueIdx = values.index(max(values))
            bestElem = positions[valueIdx][0]  # row

            # update children
            for i, (row, col) in enumerate(positions):
                self.root.children[row] = values[i]

            return bestElem

    def expand(self, treePolicy=None):
        self.depth += 1

        bestElem = self.traverse()
        print("bestElem", bestElem)
        tempEnv = deepcopy(self.navi)
        if treePolicy is None:
            while np.sum(tempEnv[bestElem, :]) < 5:
                action = random.choice(np.argwhere(tempEnv[bestElem, :] == 0).flatten())
                tempEnv[bestElem, action] = 1
                finalAction = bestElem * 4 + action
                print("finalAction", finalAction)
                val = self.simulate(finalAction)
                print("val", val)

    def simulate(self, finalAction):
        """
        returns a leaf value with the number of the cards (not considering external rewards)
        """
        # convert the action into a leaf index

        targetLeafVal = self.game.leafLen[self.prbIdx].flatten()
        if targetLeafVal[finalAction]:
            val = 1 / targetLeafVal[finalAction]
        else:
            val = 0
        return val

    def backprop(self, node, val):
        """ Updates reward and visit counts, propagating up the tree. """
        if node.parent is None:
            # increment the visit of the current node's
            node.N += 1
            # increment the value of current node's
            node.Q = self.UCB(node, val)
            # Calculate the new average reward
            node.Q += self.UCB(node, val)

        else:
            # increment the visit of the parent's
            node.parent.N += 1
            node.parent.Q += np.sum(node.Q) / node.parent.N
            # increment the visit of the current node's
            node.N += 1
            # increment the value of current node's
            ucb = self.UCB(node, val)
            # Calculate the new average reward
            node.Q += ucb

    def UCB(self, node, leafVal):
        if not node.parent:
            node.parent = self.root
        else:
            node.parent = node.parent

        if node.N > 0:
            ucb = leafVal + self.explorConstant * math.sqrt(math.log(node.parent.N) / node.N)
        else:
            ucb = 0
        return ucb



    # Tim Miller
    # selectedNode = self.select()  # row: element and col: action
    # selectedNodeAction = selectedNode.action
    # interRwd = self.simulate()
    # selectedNode.N[selectedNode] = selectedNode.N[selectedNode] + 1
    # selectedNode.Q[selectedNode, selectedNodeAction] = selectedNode.Q[selectedNode, selectedNodeAction] + interRwd

    # if self.elemState == 0:
    #     q_value = self.qTable[selectedNodeAction, self.elemState]
    #     delta = (1 / (self.N[selectedNode, selectedNodeAction])) * (interRwd - self.getValue())
    # self.qfunction.update(self.state, selectedNodeAction, delta)
    #
    # if self.parent != None:
    #     self.parent.back_propagate(self.reward + interRwd, self)

    #""" Simulate the outcome of an action, and return the child node """

    # MODEL-BASED
    # def get_outcome_child(self, action):
    #     # Choose one outcome based on transition probabilities
    #     (next_state, reward, done) = self.mdp.execute(self.state, action)
    #
    #     # Find the corresponding state and return if this already exists
    #     for (child, _) in self.children[action]:
    #         if next_state == child.state:
    #             return child
    #
    #     # This outcome has not occured from this state-action pair previously
    #     new_child = SingleAgentNode(
    #         self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
    #     )
    #
    #     # Find the probability of this outcome (only possible for model-based) for visualising tree
    #     probability = 0.0
    #     for (outcome, probability) in self.mdp.get_transitions(self.state, action):
    #         if outcome == next_state:
    #             self.children[action] += [(new_child, probability)]
    #             return new_child
