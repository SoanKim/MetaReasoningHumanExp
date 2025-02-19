#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 19:50 on 30/1/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

import pandas as pd
import numpy as np
import itertools

combi = sorted([''.join(ii) for ii in itertools.combinations([str(i) for i in range(1, 6)], r=3)])

exp1Dir = '/Users/soankim/Documents/MATLAB/Meta/new_exp1.csv'
exp2Dir = '/Users/soankim/Documents/MATLAB/Meta/new_exp2.csv'

df1 = pd.read_csv(exp1Dir, index_col=0)
df2 = pd.read_csv(exp2Dir, index_col=0)


def idxCoding(df):
    choIdxLi = []
    ansIdxLi = []
    for i, row in df.iterrows():
        choIdx = combi.index(str(row['problem_solving']))
        choIdxLi.append(choIdx)

        ansIdx = combi.index(str(row['correct_response']))
        ansIdxLi.append(ansIdx)

    df['choIdx'] = choIdxLi
    df['ansIdx'] = ansIdxLi

    return df


df1 = idxCoding(df1)
df2 = idxCoding(df2)

exp1Subj = df1.prolific_id.unique()
exp2Subj = df2.prolific_id.unique()

df1Subj1 = df1.loc[df1.prolific_id == exp1Subj[1]]  # No. 1 and 5 have all trials


def separateCards(df):
    combi = sorted([''.join(ii) for ii in itertools.combinations([str(i) for i in range(1, 6)], r=3)])
    ansM = np.zeros((df.shape[0],))
    for row in range(df.shape[0]):
        ans = df.iloc[row]['correct_response'].astype(str)
        decAns = combi.index(ans)
        ansM[row] = decAns

    one = df['one'].str.rstrip(".png").str.split("_")
    two = df['two'].str.rstrip(".png").str.split("_")
    three = df['three'].str.rstrip(".png").str.split("_")
    four = df['four'].str.rstrip(".png").str.split("_")
    five = df['five'].str.rstrip(".png").str.split("_")
    cardDf = pd.concat([one, two, three, four, five], axis=1)

    return cardDf, ansM


def digitCard(df):
    newDf, ansM = separateCards(df)
    color = ["red", "yellow", "green"]
    fill = ["open", "striped", "solid"]
    shape = ["circle", "triangle", "square"]
    back = ["black", "grey", "white"]
    attributes = [color, fill, shape, back]
    digitM = np.zeros((newDf.shape[0], 5, 4))
    for row in range(df.shape[0]):
        for five in range(5):
            for att in range(4):
                one = newDf.iloc[row, five]
                val = attributes[att].index(one[att])
                digitM[row, five, att] = val

    return [digitM, ansM]

