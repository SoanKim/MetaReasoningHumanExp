#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from createNode import Node
from treeSearch import MCTS


data = df1Subj1
scores = []
# print("len(data):", len(data))


for prbIdx in range(1):
    #monitor = np.zeros((3, 5))
    print(f"************************* new problem starts: #{prbIdx} *****************************")
    TS = MCTS(prbIdx)
    TS.traverse(node=None)
    bestChild1 = TS.getBestChild(node=None)

    TS.traverse(node=bestChild1)
    bestChild2 = TS.getBestChild(node=bestChild1)

    bestChild3 = TS.getFinalChildCurrent()
    if bestChild3 is None:
        reward = TS.selectCard(bestChild2)
    else:
        reward = TS.selectCard(bestChild3)
    print(f" ================= final reward is: {reward} ==================")








    ######## testing #######
    # depth = TS.getDepth(root)
    # print(f"depth: {depth}")
    # state, prob = TS.getState(root)
    # print(f"getState, prob: {state}, {prob}")
    # # leafVal = TS.getLeafVal(root.current)
    # # print(f"leafVal: {leafVal}")
    # print(f"is leaf? {TS.isLeaf(root)}")
    # print(f"is fullyExpanded? {TS.isFullyExpanded(root)}")
    # bestAction = TS.selectArm(root)
    # print(f"bestAction: {bestAction}")



    #print(bestChild2.__dict__)
    # TS.expand(bestChild2)

























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