#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 21:53 on 30/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from humanData import *
from collections import defaultdict

stim, label = digitCard(df1Subj1)


class qLearning:
    def __init__(self, stim, label):
        self.stim = stim
        self.label = label
        self.qTable = np.zeros((len(label), 12))
        self.buffer = defaultdict()
    def