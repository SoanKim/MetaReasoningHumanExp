#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim) at 13:57 on 1/2/25
# Title: (Enter feature name here)
# Explanation: (Enter explanation here)

from createNode import Node
import random


class MCTS:
    """
    This requires node and generates a tree.
    """

    def search(self, initialState, numIter):
        self.exploration_constant = 2
        self.root = Node(initialState, None)

        for iteration in range(numIter):
            node = self.select(self.root)
            score = self.rollout(node.table) #???
            self.backprop(node, score)

            return self.get_best_move(self.root)

    def UCB(self, child):
        """
        Calculate UCB of each child
        """
        if self.N:
            self.V += child.V / child.N
            self.V = self.V + 2 * math.sqrt(math.log(self.parent.N) / child.N)
        else:
            self.V = 0

    def select(self, node):
        """
        Tree traversal using UCB1.
        """
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

    def expand(self):
        navi = Node().move(self.prbIdx, self.current[0], self.current[1])
        actions = self.actionAvail[self.prbIdx][self.current[0]][self.current[1]]
        nextState = self.action_func(self.state, self.actionAvail)
        """s_{t+1} = (s_{t}, a_{t+1})"""
        childNode = Node(nextState)
        self.children.append(childNode)
        return self.children if len(self.children) != 0 else None


    def rollout(self, table):
        """
        random simulation
        """
        while not Game.isTerminal:
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
