#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:49 on 24/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *


class Game:
    """
    This class receives true stimuli from human data and prepares it to calculate leaf values.
    """

    def __init__(self, prbIdx=None, row=None, col=None):

        df = df1Subj1
        self.row = row if row else 0
        self.col = col if row else 0
        # self.prbM = np.zeros((len(df), 10, 3, 4))
        self.prbIdx = prbIdx if prbIdx else 0
        self.env = digitCard(df)  # (prbLen, 5, 4)
        self.stim = self.env[0]
        self.answer = self.env[1]
        self.leafLen = np.zeros((len(df), 3, 4))
        self.leafVal = np.zeros((len(df), 3, 4))
        self.cardAvail = []

        # prb and answer of one trial
        self.prb = None
        self.prb_answer = None

        # coordinates of row, column
        self._current = None

        # navigation: row: elements, columns: dimensions
        """
        You can go horizontally on the first row,
        and vertically from the second to the end rows.
        """
        self.navi = np.zeros((len(df), 3, 5))
        self.combi = [list(i) for i in itertools.combinations(list(range(5)), r=3)]

    def genLeafVal(self):
        """
        * dims:                                                    * self.leafVal:
        ColorCand: [[[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]         Same -> C,F,S,B: [[1. 1. 0. 1.] -> [1, 1, 0, 1]
        FillCand: [[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]           Err -> C,F,S,B: [9. 9. 6. 6.] -> [1/9, 1/9, 1/6, 1/6]
        ShapeCand: [[], [0, 1, 2, 4, 7, 9], [3, 5, 6, 8]]          Diff -> C,F,S,B: [0. 0. 4. 3.]] -> [0, 0, 1/4, 1/3]
        BackCand: [[4], [0, 2, 3, 5, 7, 9], [1, 6, 8]]]
        """
        for prb_i, stim in enumerate(self.stim):
            dimStim = list(zip(*stim))
            dims = []
            prbCand = [[], [], []]
            for dim in range(4):
                three = dimStim[dim]
                elemCand = [[], [], []]
                for combi_i, c in enumerate(self.combi):
                    vals = [three[cc] for cc in c]
                    lenVal = len(set(vals)) - 1
                    elemCand[lenVal].append(combi_i)
                    # self.prbM[prb_i, combi_i, lenVal, dim] = 1
                dims.append(elemCand)
                for elem in range(3):
                    prbCand[elem].append(elemCand[elem])
            self.cardAvail.append(prbCand)

            for dim in range(4):
                for elem in range(3):
                    self.leafVal[prb_i, elem, dim] = 0 if len(dims[dim][elem]) == 0 else np.round(
                        1 / len(dims[dim][elem]), 2)
                    self.leafLen[prb_i, elem, dim] = len(dims[dim][elem])
                    # np.sum(length[prb_i], axis=0) = [10. 10. 10. 10.]
        return self.leafVal

    def initPrb(self):
        # _actionAvail: all possible actions of each state
        self.leafVal = self.genLeafVal()
        # stim and answer are for each trial
        self.prb = self.stim[self.prbIdx]  # (3 * 4)
        self.prb_answer = self.answer[self.prbIdx]
        return self.prb, self.prb_answer

    # Is this for a node?
    def move(self):  # row, col from ucb select
        self.navi[self.prbIdx, self.row, self.col] = 1
        return self.navi

    def is_win(self, finalChoice):
        if finalChoice == self.answer[self.prbIdx]:
            return True

    def is_lose(self, finalChoice):
        if finalChoice != self.answer[self.prbIdx]:
            return True

    def is_terminal(self):
        if self.is_win or self.is_lose:
            return True

    def generate_states(self):
        positionAvail = []
        for row in range(3):
            for col in range(4):
                if self.navi[row, col] == 0:
                    positionAvail.append(self.move())

        return positionAvail

    def getReward(self):
        if self.is_win:
            rwd = 1 * self.leafVal[self.prbIdx]
        else:
            rwd = 0 * self.leafVal[self.prbIdx]
        return rwd

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
