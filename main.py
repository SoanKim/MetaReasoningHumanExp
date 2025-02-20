#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
# from createMCTS import MCTS
from createGame import Game
from createNode import Node
import random
import matplotlib.pyplot as plt

data = df1Subj1
scores = []
# print("len(data):", len(data))
# total = np.zeros((3, 5))

for prbIdx in range(3):
    print("************************* new problem starts *****************************")
    root = Node(prbIdx=prbIdx, current=None, traverse=False, parent=None)
    action = root.select()
    reward = root.rollout()
    root.backprop(reward=reward)
    print("root.ucbTable", root.ucbTable)

    # root.select()
    # root.expand()
    # root.rollout()
    # root.backprop()


    # root.expand(parent=root)
    #leafVal = root.expand(parent=root, heuristic=True)


        #root.backprop()
    #newNode = bestChild.select(policy=True)

    # # Check the total number of cases
    # game = Game(prbIdx=prbIdx)
    # contextM, cardAvail, answer, navi, leafState = game.prbInit()
    #

#     for row in range(contextM.shape[0]):
#         for col in range(contextM.shape[1]):
#             total[row][col] += contextM[row][col]
#
# print("total")
# print(total)
# normalized = total[:, 0] / total[:, 0].sum()
# print(normalized)





#     if timeStep == 0: --> replace it with nodeid
    #         element += random.choice(range(3))
    #         game.move(nodeID, element)
    #     else:
    #         legalMoves = game.legalMove(element)
    #         action = random.choice(legalMoves)
    #         nextState = game.move(nodeID, action)
    #
    # dimChoice = random.choice(range(4))
    # candidates = cardAvail[element][dimChoice]
    # if not candidates:
    #     finalAnswer = 0
    # else:
    #     finalAnswer = random.choice(candidates)
    # if finalAnswer == answer:
    #     score += 1
    # else:
    #     score += 0
    #
    # scores.append(np.round(score/(prbIdx+1), 2))

# plt.plot(np.arange(len(scores)), scores)
# plt.ylim([0, 1])
# plt.show()
# print(scores)