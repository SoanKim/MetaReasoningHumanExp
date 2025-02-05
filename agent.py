#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:12 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import numpy as np
from createGame import Game
from experience import replayBuffer


class Agent:
    def __init__(self):
        self.experience = replayBuffer().memory()


    # Is this for a node?
    def move(self, row=None, col=None):  # row, col from ucb select
        self.navi[self.prbIdx, row, col] = 1
        return self.navi

    def isWin(self, finalChoice):
        if finalChoice == self.answer[self.prbIdx]:
            return True

    def isLose(self, finalChoice):
        if finalChoice != self.answer[self.prbIdx]:
            return True

    def isTerminal(self):
        if self.isWin or self.isLose:
            return True

    def generateStates(self):
        positionAvail = []
        for row in range(3):
            for col in range(4):
                if self.navi[row, col] == 0:
                    positionAvail.append(self.move())

        return positionAvail

    def actionAvail(self):
        """
        self.navi = (3, 5)
        [:, 0] = visits of the first child nodes of each problem
        """
        global actions
        if np.sum(self.navi[self.prbIdx][:, 0]) < 3:
            actions = np.where(self.navi[:, 0] == 0)
        elif 3 < np.sum(self.navi[self.prbIdx][:, :]) < 15:
            cols = np.where((self.navi[:, 0] == 1))
            if len(cols) == 1:
                actions = np.where(self.navi[cols, 1:4] == 0)
            else:
                for col in cols:
                    actions = np.where(self.navi[col, 1:4] == 0)

        return actions