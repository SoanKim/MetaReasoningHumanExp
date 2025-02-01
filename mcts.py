#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from treeNode import Node
import random


class MCTS():
    def search(self, initialState, numIter):
        self.exploration_constant = 2
        self.root = Node(initialState, None)

        for iteration in range(numIter):
            node = self.select(self.root)
            score = self.rollout(node.table) #???
            self.backprop(node, score)

            return self.get_best_move(self.root)

    def select(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_move(node)

            else:
                return self.expand(node)

    def expand(self, node):
        states = node.table.generate_states()

        for state in states:
            if str(state.position) not in node.children:
                new_node = Node(state, node)
                node.children[str(state.position)] = new_node

                if len(states) == len(node.children):
                    node.is_fully_expanded = True

                return new_node
        # debugging
        print('Should not get here!!!')

    def rollout(self, table):
        while not table.is_terminal:
            try:
                table = random.choice(table.generate_states())

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
