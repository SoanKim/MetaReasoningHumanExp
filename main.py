#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)


from agent import Agent
from createGame import Game
from createMCTS import MCTS
from createNode import Node

env = Game().initPrb()
agent = Agent()
for prb_i, prb in enumerate(env):
