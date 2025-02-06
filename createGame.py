#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 10:49 on 24/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from experience import replayBuffer


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
        self.leafVal = np.zeros((len(df), 3, 4))
        self.contextM = np.zeros((len(df), 3, 5))

        # possible actions
        self.cardAvail = []

        # prb and answer of one trial
        self.prb = None
        self.prbAnswer = None

        # navigation: row: elements, columns: dimensions
        self.combi = [list(i) for i in itertools.combinations(list(range(5)), r=3)]

    def genLeafVal(self):
        """
        * dims:                                                    * self.leafVal:
        ColorCand: [[[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]         Same -> C,F,S,B: [[1. 1. 0. 1.] -> [1, 1, 0, 1]
        FillCand: [[8], [0, 1, 2, 3, 4, 5, 6, 7, 9], []]           Err -> C,F,S,B: [9. 9. 6. 6.] -> [1/9, 1/9, 1/6, 1/6]
        ShapeCand: [[], [0, 1, 2, 4, 7, 9], [3, 5, 6, 8]]          Diff -> C,F,S,B: [0. 0. 4. 3.]] -> [0, 0, 1/4, 1/3]
        BackCand: [[4], [0, 2, 3, 5, 7, 9], [1, 6, 8]]]

        * self.context
        [[ 7.  4.  1.  1.  1.]
        [24.  6.  6.  6.  6.]
        [ 9.  0.  3.  3.  3.]]
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
                    self.contextM[prb_i, :, 0] = np.sum(self.leafLen[prb_i], axis=1)
                    self.contextM[prb_i, :, 1:] = self.leafLen[prb_i]
        print(self.contextM[0])
        print(np.argwhere(self.contextM[0]))
        print(self.contextM[0][1, 0])
        # return self.cardAvail[self.prbIdx], self.leafVal[self.prbIdx]

    def initPrb(self):
        """
        initialize each prb and the navi
        """
        # stim and answer are for each trial
        self.prb = self.stim[self.prbIdx]  # (3 * 4)
        self.prbAnswer = self.answer[self.prbIdx]

        return self.prb, self.prbAnswer

    def genPrbMat(self):
        self.prb, self.prbAnswer = self.initPrb()


if __name__ == "__main__":
    game = Game()
    #prb, ans = game.initPrb()
    cards, val = game.genLeafVal()

    # print(type(cards))
    # print(type(val))