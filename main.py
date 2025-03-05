#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 16:24 on 5/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from treeSearch import MCTS
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import text

data = df1Subj1

num_agents = 2
num_trials = len(data)
avgScores = []


startTime = time.time()
print("start_time:", startTime)
totalScores = []
score = 0
for agent in range(num_agents):
    scores = np.zeros((num_agents, 4))
    for prbIdx in range(4):
        print(f"************************* new problem starts: #{prbIdx} *****************************")
        TS = MCTS(prbIdx)
        TS.traverse(node=None)
        bestChild1 = TS.getBestChild(node=None)
        print(f"bestChild'sCurrent: {bestChild1.current}")
        finalReward = TS.traverse(node=bestChild1)

        endTime = time.time()
        print(f'Job took for trial {prbIdx}: {round(endTime - startTime, 3)} sec')

        print(f"@@@@@@@@@@ Reward of the trial: {finalReward} @@@@@@@@@@@")
        scores[agent][prbIdx] += finalReward

        score += np.mean(scores) / (prbIdx + 1)
        totalScores.append(score)
print(totalScores)


# print(f">>>>>>>>>>>>>>>>> Total score: {scores} <<<<<<<<<<<<<<<<<<<<<")
# fig = plt.figure()
# plt.plot(all_scores)
# plt.title("Accuracy of Monte Carlo Tree Search")
# plt.axvline(60, 0,1, linestyle='--', color='r')
# text(50, 0.8, "TRAIN", rotation=0, verticalalignment='center')
# text(61, 0.8, "TEST", rotation=0, verticalalignment='center')
# plt.ylim([0, 1])
# fig.savefig('temp.png')





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

# avgScores = np.mean(scores, axis=0)


