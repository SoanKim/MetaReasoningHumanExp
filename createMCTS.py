#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from createNode import Node
from copy import deepcopy
from collections import defaultdict
import numpy as np
import random
import math

"""
Traverse is horizontal, and expand is vertical.
"""

class MCTS:
    def __init__(self, prbIdx):  # game = an indexed problem

        # initialize game
        self.prbIdx = prbIdx
        self.node = Node(self.prbIdx)

        # initialize Q table --> going on througout the game
        self.qTable = np.zeros((3, 5))

    def traverse(self):
        # to check if it's a leaf node or there are available actions
        self.node.select(heuristic=True)
            # print("bestChild: {}".format(bestChild.current))
            # newNode = self.node.expand(bestChild)
            # print("newNode: {}".format(newNode.current))
            # leafVal = self.node.rollout(newNode)
            # print("leafVal: {}".format(leafVal))





















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
