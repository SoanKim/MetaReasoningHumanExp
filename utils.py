#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 11:21 on 6/3/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from humanData import *
from celluloid import Camera

def makeGif(data, name):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    for i, t in enumerate(data):
        text(x=1, y=-0.6, s=name + ' (trial ' + str(i) + ')', rotation=0, fontsize=20, color='k')
        text(x=-0.4, y=0.1, s='same', rotation=0, fontsize=20, color='k')
        text(x=-0.4, y=1.1, s='error', rotation=0, fontsize=20, color='k')
        text(x=-0.4, y=2.1, s='diff', rotation=0, fontsize=20, color='k')

        text(x=1, y=0.1, s='C', rotation=0, fontsize=20, color='k')
        text(x=2, y=0.1, s='F', rotation=0, fontsize=20, color='k')
        text(x=3, y=0.1, s='S', rotation=0, fontsize=20, color='k')
        text(x=4, y=0.1, s='B', rotation=0, fontsize=20, color='k')

        text(x=1, y=1.1, s='C', rotation=0, fontsize=20, color='k')
        text(x=2, y=1.1, s='F', rotation=0, fontsize=20, color='k')
        text(x=3, y=1.1, s='S', rotation=0, fontsize=20, color='k')
        text(x=4, y=1.1, s='B', rotation=0, fontsize=20, color='k')

        text(x=1, y=2.1, s='C', rotation=0, fontsize=20, color='k')
        text(x=2, y=2.1, s='F', rotation=0, fontsize=20, color='k')
        text(x=3, y=2.1, s='S', rotation=0, fontsize=20, color='k')
        text(x=4, y=2.1, s='B', rotation=0, fontsize=20, color='k')

        ax.imshow(t, interpolation='none', vmin=0, vmax=1)
        plt.axis('off')
        camera.snap()
    anim = camera.animate()
    anim.save(name + '.mp4')

def annot(agent: str):
    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    plt.ylim([0, 1])
    plt.xlim([0, 90])
    plt.title("Accuracy of Choosing Triplets: "+agent, fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.axvline(60, color='gray', linestyle='--')
    text(x=52, y=0.1, s='train', rotation=0, fontsize=16, color='gray')
    text(x=62, y=0.1, s='test', rotation=0, fontsize=16, color='gray')

def plotLearning(scores, filename, x=90, window=1):
    plt.style.use('default')
    plt.figure(figsize=(10, 5))

    N = len(scores)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    annot("MCTS")
    plt.plot(np.arange(x), running_avg/x)
    plt.savefig(filename)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

window_size = 1
trial = len(list(range(91-window_size)))
exp_acc = df1.groupby(df1['trial_num'])['accuracy'].mean()
annot("Human")
plt.plot(np.arange(trial), moving_average(exp_acc, window_size))
plt.savefig('[Exp1.] human_accuracy.png')


# Plotting human data
#plotLearning(df1Subj1['accuracy'], '[Exp1.] human_accuracy.png', window=1)
