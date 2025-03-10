#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from treeSearch import MCTS
import time
from utils import plotLearning, makeGif
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from matplotlib.pyplot import text

data = df1Subj1
scoreHistory = []
qTables = []
ucbTables = []
freqTables = []

agentNum = 2

for agent in range(agentNum):
    score = np.zeros((agentNum))
    for prbIdx in range(90):
        print(f"************************* new problem starts: #{prbIdx} *****************************")
        TS = MCTS(prbIdx)
        TS.traverse(node=None)
        bestChild1 = TS.getBestChild(node=None)
        finalReward = TS.traverse(node=bestChild1)
        print("finalReward", finalReward)

        score[agent] += finalReward
        scoreHistory.append(score)
        qTables.append(MCTS.qTable.copy())
        ucbTables.append(TS.ucbTable.copy())
        freqTables.append(TS.freqTable.copy())


# makeGif(qTables, "Q")
# makeGif(ucbTables, "UCB")
# makeGif(ucbTables, "Visits")


# print("avgScoreHistory", scoreHistory)
# fileName = 'MCTSresult.png'
# plotLearning(scoreHistory, filename=fileName, x=90)


# startTime = time.time()
# #print("start_time:", startTime)
# endTime = time.time()
# print(f'Job took for trial {prbIdx}: {round(endTime - startTime, 3)} sec')