#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from createMCTS import MCTS
from createGame import Game
from createNode import Node
import random
import matplotlib.pyplot as plt

# from agent import Agent
# from createMCTS import MCTS
# from createNode import Node

data = df1Subj1
scores = []


for prb_i in range(1):
    score = 0
    game = Game(prb_i)
    mcts = MCTS(prb_i)

    # initialize game
    contextM, cardAvail, answer, navi = game.prbInit()
    root = Node(navi)
    element = random.choice(np.arange(3))
    for search in range(5):
        node = root
        timeStep = node.timeStep

        actions = []  # to keep track of element, the first action
        while node.isFullyExpanded(element):
            actionAvail = game.legalMove(timeStep, element)
            # Correct it later
            action = random.choice(actionAvail)
            actions.append(action)

        isTerminal = game.isTerminal(element)
        finalActionAvail = cardAvail[actions[0] * actions[-1]]

        if not isTerminal:
            node = mcts.expand(node.depth, element)
            value = mcts.simulate(element)
            node.backprop(reward=value)
            node.depthUpdate(element)

            node.children[timeStep].append(game.legalMove(timeStep, element))


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
    # scores.append(np.round(score/(prb_i+1), 2))

# plt.plot(np.arange(len(scores)), scores)
# plt.ylim([0, 1])
# plt.show()
# print(scores)