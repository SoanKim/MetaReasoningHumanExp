#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:49 on 24/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *

"""
nodeID --> 0 ... 17
Qtable --> 0 ... 17
"""


class Game:
    """
    This class receives true stimuli from human data and prepares it to calculate leaf values.
    """

    def __init__(self, prbIdx=None):
        df = df1Subj1
        self.prbIdx = prbIdx if prbIdx else 0
        self.env = digitCard(df)  # (prbLen, 5, 4)
        self.stim = self.env[0]
        self.answer = self.env[1]
        self.leafLen = np.zeros((len(df), 3, 4))

        self.contextM = np.zeros((len(df), 3, 5))
        self.navi = np.zeros_like(self.contextM)

        # It is the final state right before receiving rwd at the end of the context matrix.
        self.cardAvail = []  # list of len(3): [[[], [], [], []], [[], [], [], []], [[], [], [], []]]

        # prb and answer of one trial
        self.prb = None
        self.prbAnswer = None

        # navigation: row: elements, columns: dimensions
        self.combi = [list(i) for i in itertools.combinations(list(range(5)), r=3)]

        # reward
        self.terminalState = np.zeros((len(df), 12))

    def prbInit(self):
        """
        * dims:                                                * self.leafVal:
        ColorCand: [[[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]       Same -> C,F,S,B: [[1. 1. 0. 1.] -> [1, 1, 0, 1]
        FillCand: [[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]         Err -> C,F,S,B: [9. 9. 6. 6.] -> [1/9, 1/9, 1/6, 1/6]
        ShapeCand: [[], [0, 1, 2, 4, 7, 9], [3, 5, 6, 8]]        Diff -> C,F,S,B: [0. 0. 4. 3.]] -> [0, 0, 1/4, 1/3]
        BackCand: [[4], [0, 2, 3, 5, 7, 9], [1, 6, 8]]]

        * self.context
        [[ 7.  4.  1.  1.  1.]
        [24.  6.  6.  6.  6.]
        [ 9.  0.  3.  3.  3.]]
        """
        for prbIdx, stim in enumerate(self.stim):
            dimStim = list(zip(*stim))
            dims = []  # correct
            prbCand = [[], [], []]
            for dim in range(4):
                three = dimStim[dim]
                elemCand = [[], [], []]
                for combi_i, c in enumerate(self.combi):
                    vals = [three[cc] for cc in c]
                    lenVal = len(set(vals)) - 1
                    elemCand[lenVal].append(combi_i)

                dims.append(elemCand)
                for elem in range(3):
                    prbCand[elem].append(elemCand[elem])
            self.cardAvail.append(prbCand)  # correct

            # coding if answer is in the leaf
            temp = np.array([])
            ans = self.answer[prbIdx]
            for cards in self.cardAvail[prbIdx]:
                for elem in cards:  # list
                    if ans in elem:
                        temp = np.append(temp, 1)
                    else:
                        temp = np.append(temp, 0)

            self.terminalState[prbIdx] = temp

            # CHECK FROM HERE
            for dim in range(4):
                for elem in range(3):
                    self.leafLen[prbIdx, elem, dim] = len(dims[dim][elem])
                    self.contextM[prbIdx, :, 0] = np.sum(self.leafLen[prbIdx], axis=1)
                    self.contextM[prbIdx, :, 1:] = self.leafLen[prbIdx]

        return (self.contextM[self.prbIdx], self.cardAvail[self.prbIdx], self.answer[self.prbIdx],
                self.navi[self.prbIdx], self.terminalState[self.prbIdx])

    def legalMove(self, element=None):
        if not element:
            positions = np.argwhere(self.navi[self.prbIdx, :, 0] == 0)
        else:
            positions = np.argwhere(self.navi[self.prbIdx, element, :] == 0)
        positionAvail = positions.flatten()
        return positionAvail

    def move(self, depth, element, action=None):
        positionAvail = self.legalMove(element)
        # track state depth s0: root -> s1: child > s2: leaf
        if depth == 0:  # if any child was chosen on the root
            self.navi[self.prbIdx, action, element] = 1
        # if depth > 0
        else:
            if action in positionAvail:
                if np.sum(self.navi[self.prbIdx, element, :]) < 5:
                    self.navi[self.prbIdx, element, action + 1] = 1
            else:
                raise ValueError("Illegal move!")

    def isTerminal(self,):
        """
        To check if one node is fully expanded.
        """
        return np.sum(self.navi[self.prbIdx]) == 15

    def getReward(self, element, finalChoice):
        if finalChoice == self.answer[self.prbIdx]:
            rwd = 1
        else:
            rwd = 0
        reward = self.leafLen[self.prbIdx, element, finalChoice] * rwd
        return reward


# if __name__ == "__main__":
#     game = Game(prbIdx=0)
#     contextM, cardAvail, answer, navi, terminalVal = game.prbInit()




