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


class MCTS:

    def __init__(self, prbIdx):  # game = an indexed problem
        self.exploreConstant = 2
        # initialize game
        self.prbIdx = prbIdx
        self.game = Game(self.prbIdx)
        self.contextM, self.cardAvail, self.answer, self.navi, self.leafState = self.game.prbInit()

    def traverse(self, root=None):
        # tree traversal is for the root
        if root is None:
            root = Node(self.prbIdx)
        else:
            root = root
        root.rollout()

    def select(self, root=None):
        if root is None:
            root = Node(self.prbIdx)
        else:
            root = root
        maxChild = root.select()
        print("maxChild", maxChild)
        return maxChild

    # def expand(self, maxChild):
    #     nextState =
















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
