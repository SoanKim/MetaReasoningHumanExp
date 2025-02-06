#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from createGame import Game
import random
import matplotlib.pyplot as plt

# from agent import Agent
# from createMCTS import MCTS
# from createNode import Node

data = df1Subj1

scoreForEveryone = np.zeros((10, 89))

for participants in range(10):
    totalScore = []
    print("participant: {}".format(participants))
    for prb_i in range(89):
        game = Game(prb_i)
        contextM, cardAvail, answer, navi = game.prbInit()

        element = 0
        for timeStep in range(5):
            if timeStep == 0:
                element += random.choice(range(3))
                game.move(timeStep, element)
            else:
                legalMoves = game.legalMove(element)
                action = random.choice(legalMoves)
                nextState = game.move(timeStep, action)

        dimChoice = random.choice(range(4))
        candidates = cardAvail[element][dimChoice]
        if not candidates:
            finalAnswer = 0
        else:
            finalAnswer = random.choice(candidates)
        if finalAnswer == answer:
            totalScore.append(1)
        else:
            totalScore.append(0)

        scoreForEveryone[participants, prb_i] = np.mean(totalScore)


plt.plot(np.mean(scoreForEveryone, axis=0))
plt.show()
# plt.title("random problem solving")
# plt.plot(list(range(89)), scoreForPlotting)
# plt.show()