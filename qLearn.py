#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 21:53 on 30/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from collections import defaultdict
from createNode import node

stim, label = digitCard(df1Subj1)


class learning:
    def __init__(self, state, actFunc=None):
        self.state = state
        self.buffer = defaultdict()
        self.actFunc = actFunc

    def chooseAction(self):
        actionAvail = validateAction(self.state)
        if self.actFunc:
            action = self.actFunc(self.state)
        else:
            action = UCB(self.state)

    def nextState(self):
        """s_{t+1} = (s_{t}, a_{t+1})"""
        nextS = self.state[self.action]
        return node(nextS)

    def train(self):
        for t in range(len(self.stim)):
            obs = self.stim[t]
            elements = 