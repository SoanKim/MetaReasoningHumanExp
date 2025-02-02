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

    def __init__(self, df=None, prbIdx=None):

        df = df1Subj1 if None else df
        # self.prbM = np.zeros((len(df), 10, 3, 4))
        self.prbIdx = prbIdx if prbIdx else 0
        self.actionAvail = []
        self.env = digitCard(df)  # (prbLen, 5, 4)
        self.stim, self.answer = self.env
        self.leafLen = np.zeros((len(df), 3, 4))
        self.leafVal = np.zeros((len(df), 3, 4))

        # prb and answer of one trial
        self.prb = None
        self.prb_answer = None

        # coordinates of row, column
        self._current = None

        # navigation: row: elements, columns: dimensions
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
        for prb_i, (stim, ans) in enumerate(self.env):
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
            self.actionAvail.append(prbCand)

            for dim in range(4):
                for elem in range(3):
                    self.leafVal[prb_i, elem, dim] = 0 if len(dims[dim][elem]) == 0 else np.round(
                        1 / len(dims[dim][elem]), 2)
                    self.leafLen[prb_i, elem, dim] = len(dims[dim][elem])
                    # np.sum(length[prb_i], axis=0) = [10. 10. 10. 10.]
        return self.actionAvail, self.env, self.leafLen, self.leafVal

    def initPrb(self):
        # _actionAvail: all possible actions of each state
        self.actionAvail, self.env, self.leafLen, self.leafVal = Game().genLeafVal()
        # stim and answer are for each trial
        stim, answer = self.env
        self.prb = stim[self.prbIdx]  # (3 * 4)
        self.prb_answer = answer[self.prbIdx]

    # Is this for a node?
    def move(self, row, col):  # row, col from ucb select
        self.navi[self.prbIdx, row, col] = 1
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

    def generate_states(self, row, col):
        positionAvail = []
        for row in range(3):
            for col in range(4):
                if self.navi[row, col] == 0:
                    positionAvail.append(self.move(row, col))

        return positionAvail

    def getReward(self, prbIdx):
        if self.is_terminal:
            if self.is_win:
                reward = 1 + self.leafVal[prbIdx]

    def actionAvail(self):
        """
        self.navi = (3, 5)
        [:, 0] = visits of the first child nodes of each problem
        """
        if np.sum(self.navi[self.prbIdx][:, 0]) <3:
            actions = np.arange(3)
        elif 3 < np.sum(self.navi[self.prbIdx][:, :]) < 15:
            actions = np.arange(4)
        else:
            actions = self.actionAvail

        return actions



# makeTree().initState()
# def UCB(self, child):
#     if self.N:
#         self.V += child.V / child.N
#         self.V = self.V + 2 * math.sqrt(math.log(self.parent.N) / child.N)
#     else:
#         self.V = 0



#

#
# def selectNode(self):
#
#     while len(self.children) > 0 :
#         children = self.children.V
#         max_value = max(children, key=lambda x: x.value()).value()
#         max_nodes = [n for n in children if n.value()==max_value]
#
#         node = np.random.choice(max_nodes)
#         root_state.move(node.move)
#
#     bestChild = None
#     bestUCB = -np.inf
#
#     for child in self.children:
#         ucb = self.UCB(child)
#         if ucb > bestUCB:
#             bestChild = child
#             bestUCB = ucb
#     return bestChild

# if __name__ == "__main__":
#     leaf, length, action = makeTree().genLeaf()
#     for prb in range(len(leaf)):
#         print(" ########### prb {} ##########".format(prb))
#         for elem in range(3):
#             print("new dim {}".format(elem))
#             for dim in range(4):
#                 print(action[prb][elem][dim])
