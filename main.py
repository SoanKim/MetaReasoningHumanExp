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

score = 0
for prbIdx in range(90):
    print(f"************************* new problem starts: #{prbIdx} *****************************")
    TS = MCTS(prbIdx)
    TS.traverse(node=None)
    bestChild1 = TS.getBestChild(node=None)
    finalReward = TS.traverse(node=bestChild1)
    print("finalReward", finalReward)

    score += finalReward
    scoreHistory.append(score)
    qTables.append(MCTS.qTable.copy())
    ucbTables.append(TS.ucbTable.copy())
    freqTables.append(TS.freqTable.copy())


makeGif(qTables, "Q")
makeGif(ucbTables, "UCB")
makeGif(ucbTables, "Visits")

# imgs = [Image.matshow(img) for img in qTables]
# # duration is the number of milliseconds between frames; this is 40 frames per second
# imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)

print("avgScoreHistory", scoreHistory)
fileName = 'MCTSresult.png'
plotLearning(scoreHistory, filename=fileName, x=90)


# startTime = time.time()
# #print("start_time:", startTime)
# endTime = time.time()
# print(f'Job took for trial {prbIdx}: {round(endTime - startTime, 3)} sec')