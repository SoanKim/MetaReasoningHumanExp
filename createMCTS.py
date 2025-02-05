#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createGame import *
from createNode import Node
from copy import deepcopy
import numpy as np
import random
import math


class MCTS:
    """
    This requires node and generates a tree.
    Select --> Expand --> Simulate --> Backup
    """
    def __init__(self, game, parent=None, policy=None, exploration_constant=2):  # game is the problem with index
        self.exploration_constant = exploration_constant
        self.game = game
        self.policy = policy
        self.root = Node(game)  # init problem
        self.node = deepcopy(self.root)
        self.parent = parent

    def UCB(self, child):
        """
        Calculate UCB of each child
        """
        if child.N > 0:
            child.Q += child.V / child.N
            ucb = child.Q + 2 * math.sqrt(math.log(self.parent.N) / child.N)
        else:
            ucb = 0
        return ucb

    def selectNode(self):  # Should select from the navi position
        """
        Tree traversal using either mere valueUCB1.
        """
        bestChild = []  # It can be plural
        bestUCB = -np.inf

        for child in self.node.children:
            ucb = self.UCB(child)
            if ucb > bestUCB:
                bestChild.append(child)
                bestUCB += ucb

        # In case bestChild is plural
        bestChild = random.choice(bestChild)
        self.node.memory['bestChild'] = bestChild
        self.node.memory['bestUCB'] = bestUCB

        return bestChild

    def expand(self):
        # https://joshvarty.github.io/AlphaZero/
        while len(self.node.children()) > 0:
            action = self.selectNode()
            self.searchPath.append(action)

            parent = self.searchPath[-2]
            # parent.state: coordinates
            actionAvail = self.game.generateStates(row=parent.state[0], col=parent.state[1])
            action = self.selectNode()
            nextState = parent.state, action

            if value is None:
                # EXPAND
                value = self.node.move(parent.state, nextState)

            self.backup(self.searchPath, value)

    #     for state in states:
    #         if str(state.position) not in node.children:
    #             new_node = Node(state, node)
    #             node.children[str(state.position)] = new_node
    #
    #             if len(states) == len(node.children):
    #                 node.is_fully_expanded = True
    #
    #             return new_node
    #     # debugging
    #     print('Should not get here!!!')
    #
    # def expand(self):
    #     navi = Node().move(self.prbIdx, self.current[0], self.current[1])
    #     actions = self.actionAvail[self.prbIdx][self.current[0]][self.current[1]]
    #     nextState = self.action_func(self.state, self.actionAvail)
    #     """s_{t+1} = (s_{t}, a_{t+1})"""
    #     childNode = Node(nextState)
    #     self.children.append(childNode)
    #     return self.children if len(self.children) != 0 else None

    def rollout(self, table):
        """
        random simulation
        """
        while not Game.isTerminal:
            try:
                table = random.choice(table.generateStates())

            except:
                return 0

    def backprop(self, node, score):
        while node is not None:
            node.visits += 1
            node.score += score
            node = node.parent

    def get_best_move(self, node):
        best_score = float('-inf')
        best_moves = []

        for child_node in node.children.values():
            move_score = child_node.score / child_node.score / child_node.visits + self.exploration_constant

            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]

            elif move_score == best_score:
                best_moves.append(child_node)

        return random.choice(best_moves)
