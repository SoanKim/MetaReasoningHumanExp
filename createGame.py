#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:04 on 10/3/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *


class Game:
    """
    This class receives true stimuli from human data and prepares it to calculate leaf values by the number of cases.
    """

    def __init__(self, prbIdx):
        df = df1Subj1
        self.prbIdx = prbIdx
        self.env = digitCard(df)
        self.stim = self.env[0]
        self.answer = self.env[1]
        self.leafLen = np.zeros((len(df), 4, 3))
        self.contextM = np.zeros((len(df), 4, 4))

        # It is the final state right before receiving rwd at the end of the context matrix.
        self.cardAvail = []  # list of len(4): [[[], [], []],/n [[], [], []],/n [[], [], []],/n [[], [], []]]

        # prb and answer of one trial
        self.prb = None
        self.prbAnswer = None

        # navigation: row: elements, columns: dimensions
        self.combi = sorted([list(i) for i in itertools.combinations(list(range(5)), r=3)])

        # reward
        self.terminalState = np.zeros((len(df), 12))

    def prbInit(self):
        for prbIdx, stim in enumerate(self.stim):
            dimStim = list(zip(*stim))  # "*" unpacks a list (or other iterable), making each of its elements separate.
            dims = []  # correct. 4 dims

            prbCand = [[], [], []]  # In each [], 4 list of combi_i regardless of dims
            for dim in range(4):
                groupByDim = dimStim[dim]
                elemCand = [[], [], []]  # unique dim values ([0]: all the same, [1]: error, [2]: all different)
                for combi_i, c in enumerate(self.combi):  # 012, 013, ..., 234
                    vals = [groupByDim[cc] for cc in c]  # group each dimension by random three picks.
                    lenVal = len(set(vals)) - 1  # subtract 1 because it will be positional indices.
                    elemCand[lenVal].append(combi_i)  # tentative triplets

                dims.append(elemCand)  # four dims
                for elem in range(3):
                    prbCand[elem].append(elemCand[elem])  # append all tentative triplets regardless of dims

                    # CHECK FROM HERE
                    self.leafLen[prbIdx, dim, elem] = len(dims[dim][elem])

                    self.contextM[prbIdx, :, 0] = np.sum(self.leafLen[prbIdx], axis=1)
                    self.contextM[prbIdx, :, 1:] = self.leafLen[prbIdx]
                    self.cardAvail.append(dims)  # correct

            temp = np.array([])
            ans = self.answer[prbIdx]
            for cards in self.cardAvail[prbIdx]:
                for elem in cards:  # list
                    if ans in elem:
                        temp = np.append(temp, 1)
                    else:
                        temp = np.append(temp, 0)

            self.terminalState[prbIdx] = temp  # correct

        return (self.contextM[self.prbIdx], self.cardAvail[self.prbIdx],
                self.answer[self.prbIdx], self.terminalState[self.prbIdx])
