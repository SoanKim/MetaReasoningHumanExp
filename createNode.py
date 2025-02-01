#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 18:57 on 31/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from envStructure import *


class node:
    def __init__(self, prbIdx, state=None, action=None, parent=None, actionFunc = None):
        self.prbIdx = prbIdx if prbIdx else 0
        self.actionAvail, self.env, self.leafLen, self.leafVal = Table().genLeafVal()
        self.N = 0
        self.Q = 0

        self.children = []
        stim, answer = self.env
        self.root = stim[self.prbIdx]  # (3 * 4)
        self.answer = answer[self.prbIdx]

        self.state = state if state else self.root
        self.action = action
        self.parent = parent
        self.action_func = actionFunc if actionFunc else self.UCB

    def hasChildren(self):
        """Check if visited or not"""
        return len(self.children) > 0

    def expand(self):
        nextState = self.action_func(self.state, self.actionAvail)
        """s_{t+1} = (s_{t}, a_{t+1})"""
        childNode = node(nextState)
        self.children.append(childNode)
        return self.children if len(self.children) != 0 else None

    def bestChild(self):
        if not self.children:
            # No children, so no best child
            return None

            # Apply policy function to select the best child
        return np.argmax(self.children)

    def getRwd(self):
        if self.action == self.answer:
            rwd = 1
        else:
            rwd = 0
        self.reward = self.leafVal[self.action] + rwd
        return self.reward

    def backprop(self, reward):
        """ Updates reward and visit counts, propagating up the tree. """
        # Increment the number of visits
        self.N += 1

        # Calculate the new average reward
        self.Q += (reward - self.reward) / self.visits

        # Propagate reward to the parent
        if self.parent:
            self.parent.backpropagate(reward)