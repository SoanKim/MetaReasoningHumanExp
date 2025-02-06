#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from agent import Agent
from createGame import Game
from createMCTS import MCTS
from createNode import Node

data = df1Subj1
# agent = Agent()
for prb_i in range(len(data)):
    states, actions, answer = Game(prb_i).prbInit()
    parentAction =
    print("####### prb_i: {} #######:".format(prb_i))
    # print("states:", states)
    # print("actions", actions)
    # print("answer:", answer)
    # print(np.zeros_like(states))