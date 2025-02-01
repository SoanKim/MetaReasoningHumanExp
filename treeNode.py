#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:32 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

class Node():
    def __init__(self, game, parent):
        self.game = game

        if self.game.is_win() or self.game.is_lose():
            self.is_terminal = True

        else:
            self.is_terminal = False

        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.visits = 0
        self.score = 0
        self.children = {}

        # define table