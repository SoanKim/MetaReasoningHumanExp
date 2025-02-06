#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from createGame import Game
import random

# from agent import Agent
# from createMCTS import MCTS
# from createNode import Node

data = df1Subj1
# agent = Agent()
for prb_i in range(1):
    game = Game(prb_i)
    contextM, cardAvail, answer, navi = game.prbInit()

    element = 0
    for timeStep in range(5):
        if timeStep == 0:
            element += random.choice(range(3))
            game.move(timeStep, element)
            print("element, {}".format(element))
        else:
            legalMoves = game.legalMove(element)
            print("legal moves: {}".format(legalMoves))
            action = random.choice(legalMoves)
            nextState = game.move(timeStep, action)
            print("action, {}".format(action))


